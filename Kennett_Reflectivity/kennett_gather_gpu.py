"""GPU-accelerated seismogram gather via Kennett reflectivity.

Architecture:
    1. Kennett recursion: batched NumPy in float64 (kennett_reflectivity_batch).
       Runs on CPU — only 4 interface steps, needs double precision for
       evanescent phase factors that underflow in float32.

    2. Bessel-weighted wavenumber summation: GPU via PyTorch (float32-safe).
       This is the true bottleneck (np_slow × nr × nfreq operations).
       The Bessel function J₀ and reflectivity amplitudes are O(1),
       so float32 is perfectly adequate here.

    3. IFFT: CPU via NumPy (fast, per-trace).

References
----------
Bouchon, M. (1981). A simple method to calculate Green's functions
for elastic layered media. BSSA, 71(4), 959-971.
"""

from __future__ import annotations

import logging
import time as time_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import collections.abc
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .kennett_reflectivity_gpu import get_device, kennett_reflectivity_batch
from .layer_model import LayerModel
from .source import ricker_spectrum

__all__ = ["compute_gather_gpu", "default_ocean_crust_model"]

logger = logging.getLogger(__name__)


def default_ocean_crust_model() -> LayerModel:
    """5-layer ocean-crust model (same as kennett_seismogram)."""
    return LayerModel.from_arrays(
        alpha=[1.5, 1.6, 3.0, 5.0, 2.2],
        beta=[0.0, 0.3, 1.5, 3.0, 1.1],
        rho=[1.0, 2.0, 3.0, 3.0, 1.8],
        thickness=[2.0, 1.0, 1.0, 1.0, np.inf],
        Q_alpha=[20000, 100, 100, 100, 100],
        Q_beta=[1e10, 100, 100, 100, 100],
    )


def _gpu_bessel_summation(
    R: np.ndarray,
    p_samples: np.ndarray,
    omega_real: np.ndarray,
    offsets: np.ndarray,
    dp: float,
    device: torch.device,
) -> np.ndarray:
    """GPU-accelerated Bessel-weighted wavenumber summation.

    Computes U[r, w] = dp · Σ_j R[j,w] · J₀(ω_w · p_j · r) · p_j · ω_w

    All values here are O(1) in magnitude, so float32 is safe.
    On MPS this gives significant speedup over the CPU einsum.

    Parameters
    ----------
    R : np.ndarray, complex128, shape (np_slow, nfreq)
        Reflectivity from batched Kennett computation.
    p_samples : np.ndarray, float64, shape (np_slow,)
        Slowness samples.
    omega_real : np.ndarray, float64, shape (nfreq,)
        Real angular frequencies.
    offsets : np.ndarray, float64, shape (nr,)
        Receiver offsets in km.
    dp : float
        Slowness spacing.
    device : torch.device
        PyTorch device for GPU computation.

    Returns
    -------
    np.ndarray, complex128, shape (nr, nfreq)
        Accumulated spectral displacement U(r, ω).
    """
    np_slow = len(p_samples)
    nfreq = len(omega_real)
    nr = len(offsets)

    # Hann taper on slowness to suppress truncation artefacts (Gibbs ringing).
    # Without tapering, the abrupt cutoff at p_max produces evenly-spaced
    # ripples in the time domain. The cosine taper smoothly reduces the
    # contribution to zero at p_max, eliminating these artefacts at the
    # cost of slightly reduced resolution for near-critical phases.
    p_max = p_samples[-1]
    taper = 0.5 * (1.0 + np.cos(np.pi * p_samples / p_max))  # (np_slow,)

    # Weighted reflectivity: W[j, w] = R[j,w] · taper[j] · p_j · ω_w · dp
    weight = (
        taper[:, np.newaxis] * p_samples[:, np.newaxis] * omega_real[np.newaxis, :] * dp
    )
    W = R * weight  # (np_slow, nfreq), complex128

    # Accumulate U(r, ω) in chunks to control memory.
    # Each chunk processes a block of slowness samples.
    # The (nr, chunk, nfreq) Bessel array is the memory bottleneck.
    U = np.zeros((nr, nfreq), dtype=np.complex128)

    # Chunk size tuned for ~512 MB GPU memory per chunk
    # For nr=100, nfreq=2047: 100 * 256 * 2047 * 4 bytes ≈ 200 MB (float32)
    chunk_size = min(np_slow, 256)
    n_chunks = (np_slow + chunk_size - 1) // chunk_size

    # Transfer offsets and omega to GPU once
    off_t = torch.from_numpy(offsets.astype(np.float32)).to(device)  # (nr,)
    omega_t = torch.from_numpy(omega_real.astype(np.float32)).to(device)  # (nfreq,)

    chunk_iter = tqdm(
        range(0, np_slow, chunk_size),
        desc="Bessel summation (GPU)",
        unit="chunk",
        total=n_chunks,
    )
    for j_start in chunk_iter:
        j_end = min(j_start + chunk_size, np_slow)

        # Transfer this chunk's weighted reflectivity to GPU
        # Split into real and imag parts (float32 each)
        W_chunk = W[j_start:j_end]  # (n_chunk, nfreq), complex128
        W_re = torch.from_numpy(W_chunk.real.astype(np.float32)).to(device)
        W_im = torch.from_numpy(W_chunk.imag.astype(np.float32)).to(device)

        # Slowness values for this chunk
        p_chunk = torch.from_numpy(p_samples[j_start:j_end].astype(np.float32)).to(
            device
        )  # (n_chunk,)

        # Bessel arguments: arg[r, j, w] = ω_w · p_j · r_i
        # Use outer products: (nr, 1, 1) * (1, n_chunk, 1) * (1, 1, nfreq)
        arg = (
            off_t[:, None, None] * p_chunk[None, :, None] * omega_t[None, None, :]
        )  # (nr, n_chunk, nfreq)

        # Bessel J₀ via torch — use the sinc-based series or just transfer
        # to CPU for scipy. torch doesn't have j0, so compute on CPU.
        arg_cpu = arg.cpu().numpy()
        from scipy.special import j0

        J0_cpu = j0(arg_cpu).astype(np.float32)
        J0_t = torch.from_numpy(J0_cpu).to(device)  # (nr, n_chunk, nfreq)

        # GPU einsum: U_chunk[r, w] = Σ_j J0[r,j,w] · W_re[j,w]
        # (and same for imaginary part)
        U_re_chunk = torch.einsum("rjw,jw->rw", J0_t, W_re)  # (nr, nfreq)
        U_im_chunk = torch.einsum("rjw,jw->rw", J0_t, W_im)  # (nr, nfreq)

        # Transfer back to CPU and accumulate in float64
        U += U_re_chunk.cpu().numpy().astype(np.float64)
        U += 1j * U_im_chunk.cpu().numpy().astype(np.float64)

    return U


def compute_gather_gpu(
    model: LayerModel,
    offsets: np.ndarray,
    T: float = 64.0,
    nw: int = 2048,
    np_slow: int = 4096,
    p_max: float = 1.2,
    gamma: float | None = None,
    source_func: collections.abc.Callable | None = None,
    free_surface: bool = False,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a seismogram gather with GPU-accelerated Bessel summation.

    The Kennett reflectivity is computed in float64 on CPU (batched NumPy).
    The Bessel-weighted wavenumber summation is done on GPU in float32.
    The IFFT is done on CPU.

    Parameters
    ----------
    model : LayerModel
        Stratified elastic model.
    offsets : np.ndarray
        Receiver offsets in km, shape (nr,).
    T : float
        Time window in seconds.
    nw : int
        Number of positive frequencies (power of 2).
    np_slow : int
        Number of slowness samples for the integration.
    p_max : float
        Maximum slowness (s/km).
    gamma : float | None
        Complex frequency damping factor γ (rad/s). Default: π/T.
    source_func : callable | None
        Source spectrum function(omega, omega_max) -> S. Default: Ricker.
    free_surface : bool
        If True, include free surface reflections.
    device : torch.device | None
        PyTorch device for Bessel summation. Default: auto-detect.

    Returns
    -------
    time : np.ndarray, shape (nt,)
    offsets : np.ndarray, shape (nr,)
    gather : np.ndarray, shape (nr, nt)
    """
    if device is None:
        device = get_device()

    nr = len(offsets)

    # --- Frequency damping (Bouchon, 1981) ---
    if gamma is None:
        gamma = np.pi / T
    logger.info(f"Complex frequency damping: γ = {gamma:.6f} rad/s")

    # --- Frequency grid ---
    dw = 2.0 * np.pi / T
    nwm = nw - 1
    wmax = nw * dw
    omega_real = np.arange(1, nwm + 1, dtype=np.float64) * dw
    omega_damped = omega_real + 1j * gamma

    # --- Source spectrum ---
    if source_func is None:
        S = ricker_spectrum(omega_real, wmax)
    else:
        S = source_func(omega_real, wmax)

    # --- Time axis ---
    nt = 2 * nw
    time = np.arange(nt, dtype=np.float64) * (T / float(nt))

    # --- Slowness grid ---
    dp = p_max / np_slow
    p_samples = np.arange(1, np_slow + 1, dtype=np.float64) * dp

    logger.info(
        f"GPU Gather: {nr} offsets [{offsets[0]:.1f}-{offsets[-1]:.1f} km], "
        f"{nwm} frequencies, {np_slow} slowness samples, "
        f"dp={dp:.6f} s/km, p_max={p_max:.3f} s/km, device={device}"
    )

    # ===== STEP 1: Batched reflectivity (CPU, float64) =====
    t0 = time_module.perf_counter()
    R = kennett_reflectivity_batch(
        model,
        p_samples,
        omega_damped,
        free_surface=free_surface,
    )
    t1 = time_module.perf_counter()
    logger.info(f"Reflectivity computed in {t1 - t0:.1f}s")

    # ===== STEP 2: Bessel summation (GPU, float32) =====
    t0 = time_module.perf_counter()
    U = _gpu_bessel_summation(R, p_samples, omega_real, offsets, dp, device)

    # Multiply by source spectrum
    U *= S[np.newaxis, :]
    t1 = time_module.perf_counter()
    logger.info(f"Bessel summation computed in {t1 - t0:.1f}s")

    # ===== STEP 3: IFFT to time domain with damping compensation =====
    gather = np.zeros((nr, nt), dtype=np.float64)
    exp_decay = np.exp(-gamma * time)
    for ir in tqdm(range(nr), desc="IFFT → time domain", unit="trace"):
        Uwk = np.zeros(nt, dtype=np.complex128)
        Uwk[1:nw] = U[ir, :]
        Uwk[nw + 1 :] = np.conj(U[ir, ::-1])

        seismogram_c = np.fft.fft(Uwk)
        gather[ir, :] = np.real(seismogram_c) * exp_decay

    return time, offsets, gather


def main() -> None:
    """CLI entry point for GPU gather computation."""
    import argparse

    from .kennett_gather import plot_gather

    parser = argparse.ArgumentParser(
        description="GPU-accelerated seismogram gather via Kennett reflectivity.",
    )
    parser.add_argument(
        "--r-min",
        type=float,
        default=0.5,
        help="Minimum offset in km (default: 0.5)",
    )
    parser.add_argument(
        "--r-max",
        type=float,
        default=50.0,
        help="Maximum offset in km (default: 50.0)",
    )
    parser.add_argument(
        "--dr",
        type=float,
        default=0.5,
        help="Offset spacing in km (default: 0.5)",
    )
    parser.add_argument(
        "-T",
        "--duration",
        type=float,
        default=64.0,
        help="Time window in seconds (default: 64.0)",
    )
    parser.add_argument(
        "-n",
        "--nw",
        type=int,
        default=2048,
        help="Number of positive frequencies (default: 2048)",
    )
    parser.add_argument(
        "--np",
        type=int,
        default=4096,
        dest="np_slow",
        help="Number of slowness samples (default: 4096)",
    )
    parser.add_argument(
        "--p-max",
        type=float,
        default=1.2,
        help="Maximum slowness in s/km (default: 1.2)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Complex frequency damping γ in rad/s (default: π/T)",
    )
    parser.add_argument(
        "--t-max",
        type=float,
        default=30.0,
        help="Maximum time to display in seconds (default: 30.0)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="gather_gpu.png",
        help="Output plot filename (default: gather_gpu.png)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plot generation, save data only",
    )
    parser.add_argument(
        "--free-surface",
        action="store_true",
        help="Include free surface reflections (water-column multiples)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="PyTorch device: 'mps', 'cuda', or 'cpu' (default: auto-detect)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    model = default_ocean_crust_model()
    offsets = np.arange(args.r_min, args.r_max + args.dr / 2, args.dr)

    device = None
    if args.device is not None:
        device = torch.device(args.device)

    logger.info(
        f"Model: {model.n_layers} layers, "
        f"{len(offsets)} offsets [{offsets[0]:.1f}-{offsets[-1]:.1f} km]"
    )

    t_start = time_module.perf_counter()
    time_arr, offsets, gather = compute_gather_gpu(
        model,
        offsets,
        T=args.duration,
        nw=args.nw,
        np_slow=args.np_slow,
        p_max=args.p_max,
        gamma=args.gamma,
        free_surface=args.free_surface,
        device=device,
    )
    t_end = time_module.perf_counter()
    logger.info(f"Total computation time: {t_end - t_start:.1f}s")

    if not args.no_plot:
        try:
            fs_label = " + free surface" if args.free_surface else ""
            device_label = str(device or get_device())
            plot_gather(
                time_arr,
                offsets,
                gather,
                output=args.output,
                title=(
                    f"Kennett Reflectivity Gather — 5-layer ocean crust{fs_label}\n"
                    f"GPU ({device_label}), np={args.np_slow}, "
                    f"p_max={args.p_max} s/km"
                ),
                t_max=args.t_max,
            )
        except ImportError:
            logger.warning("matplotlib not available, skipping plot")

    output_npz = Path(args.output).with_suffix(".npz")
    np.savez(
        output_npz,
        time=time_arr,
        offsets=offsets,
        gather=gather,
    )
    logger.info(f"Data saved to {output_npz}")


if __name__ == "__main__":
    main()
