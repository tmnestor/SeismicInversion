"""Compute synthetic seismogram gathers using discrete wavenumber summation.

Extends the single-slowness Kennett reflectivity to the space-time domain
via the discrete wavenumber method (Bouchon, 1981). For an explosive
(isotropic) point source in a layered medium the vertical displacement
at offset r is:

    u_z(r, t) = Re{ (1/2π) ∫ dω e^{-iωt} ∫₀^∞ R(ω,k) S(ω) J₀(kr) k dk }

Substituting k = ωp, dk = ω dp gives integration over slowness:

    u_z(r, ω) = ω Δp Σⱼ R(ω, pⱼ) S(ω) J₀(ω pⱼ r) pⱼ

This formulation exploits the existing vectorisation of kennett_reflectivity
over frequency: each slowness sample requires just ONE call that processes
all frequencies simultaneously, rather than nfreq × nk scalar calls.

Complex frequency damping ω → ω + iγ (Bouchon, 1981) suppresses spatial
wrap-around from the discrete slowness periodicity.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import collections.abc
from functools import partial
from pathlib import Path

import numpy as np
from scipy.special import j0

from .kennett_reflectivity import kennett_reflectivity
from .layer_model import LayerModel
from .source import ricker_spectrum

__all__ = ["compute_gather", "default_ocean_crust_model"]

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


def _process_slowness_batch(
    jp_batch: list[int],
    p_samples: np.ndarray,
    omega_damped: np.ndarray,
    offsets: np.ndarray,
    dp: float,
    model_params: dict,
    free_surface: bool = False,
    taper: np.ndarray | None = None,
) -> list[tuple[int, np.ndarray]]:
    """Process a batch of slowness samples.

    For each slowness pⱼ, compute R(ω, pⱼ) for ALL frequencies at once
    (vectorised inside kennett_reflectivity), then form the contribution
    to the wavenumber integral for each offset.

    Parameters
    ----------
    jp_batch : list[int]
        Slowness indices to process.
    p_samples : np.ndarray
        Slowness samples, shape (np,).
    omega_damped : np.ndarray
        Complex angular frequencies (ω + iγ), shape (nwm,).
    offsets : np.ndarray
        Receiver offsets in km, shape (nr,).
    dp : float
        Slowness spacing.
    model_params : dict
        Serialised LayerModel parameters.
    free_surface : bool
        If True, include free surface reflections.

    Returns
    -------
    list[tuple[int, np.ndarray]]
        List of (slowness_index, contribution) where contribution
        has shape (nr, nwm) — the additive contribution to U(r, ω).
    """
    model = LayerModel.from_arrays(**model_params)
    omega_real = omega_damped.real  # shape (nwm,)
    results = []

    for jp in jp_batch:
        pj = p_samples[jp]

        # Reflectivity for ALL frequencies at this slowness (vectorised!)
        R = kennett_reflectivity(
            model,
            float(pj),
            omega_damped,
            free_surface=free_surface,
        )  # (nwm,)

        # J₀(ω pⱼ rᵢ) for each offset and frequency: shape (nr, nwm)
        # argument = omega_real * pj * r  -> (nr, nwm)
        arg = omega_real[np.newaxis, :] * pj * offsets[:, np.newaxis]
        J0_vals = j0(arg)  # (nr, nwm)

        # Contribution: R(ω, p) * taper * pⱼ * ω * dp * J₀(ωpr)
        # weight = taper[jp] * pj * omega_real * dp  -> (nwm,)
        tap = taper[jp] if taper is not None else 1.0
        weight = tap * pj * omega_real * dp  # (nwm,)
        contrib = J0_vals * (R * weight)[np.newaxis, :]  # (nr, nwm)

        results.append((jp, contrib))

    return results


def compute_gather(
    model: LayerModel,
    offsets: np.ndarray,
    T: float = 64.0,
    nw: int = 2048,
    np_slow: int = 2048,
    p_max: float = 1.0,
    gamma: float | None = None,
    source_func: collections.abc.Callable | None = None,
    n_workers: int | None = None,
    free_surface: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a seismogram gather via discrete wavenumber summation.

    The integration is performed over slowness p rather than wavenumber k,
    exploiting the frequency-vectorised kennett_reflectivity. Each of the
    np_slow slowness samples requires just ONE reflectivity call that
    processes all nwm frequencies simultaneously.

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
    n_workers : int | None
        Number of parallel workers. Default: number of CPU cores.
    free_surface : bool
        If True, include free surface reflections (R_fs = -1) at the
        ocean surface. Generates water-column reverberations (surface
        multiples). Default: False.

    Returns
    -------
    time : np.ndarray
        Time samples, shape (nt,).
    offsets : np.ndarray
        Offsets in km, shape (nr,).
    gather : np.ndarray
        Seismogram gather, shape (nr, nt).

    Notes
    -----
    Substituting k = ωp in the Hankel transform gives:

        u(r, ω) = Δp Σⱼ R(ω+iγ, pⱼ) S(ω) J₀(ω pⱼ r) pⱼ ω

    Only np_slow reflectivity evaluations are needed (each vectorised
    over all frequencies), compared to nfreq × nk scalar calls in the
    wavenumber-loop approach.

    References
    ----------
    Bouchon, M. (1981). A simple method to calculate Green's functions
    for elastic layered media. BSSA, 71(4), 959-971.
    """
    nr = len(offsets)

    # --- Frequency damping (Bouchon, 1981) ---
    if gamma is None:
        gamma = np.pi / T
    logger.info(f"Complex frequency damping: γ = {gamma:.6f} rad/s")

    # --- Frequency grid ---
    dw = 2.0 * np.pi / T
    nwm = nw - 1
    wmax = nw * dw
    omega = np.arange(1, nwm + 1, dtype=np.float64) * dw  # (nwm,)
    omega_damped = omega + 1j * gamma  # complex frequencies

    # --- Source spectrum ---
    if source_func is None:
        S = ricker_spectrum(omega, wmax)
    else:
        S = source_func(omega, wmax)

    # --- Time axis ---
    nt = 2 * nw
    dt = T / float(nt)
    time = np.arange(nt, dtype=np.float64) * dt

    # --- Slowness grid (uniform in p) ---
    dp = p_max / np_slow
    p_samples = np.arange(1, np_slow + 1, dtype=np.float64) * dp  # skip p=0

    # Hann taper to suppress truncation artefacts at p_max
    taper = 0.5 * (1.0 + np.cos(np.pi * p_samples / p_max))  # (np_slow,)

    logger.info(
        f"Gather: {nr} offsets [{offsets[0]:.1f}-{offsets[-1]:.1f} km], "
        f"{nwm} frequencies, {np_slow} slowness samples, "
        f"dp={dp:.6f} s/km, p_max={p_max:.3f} s/km"
    )
    logger.info(
        f"Total reflectivity calls: {np_slow} (each vectorised over {nwm} frequencies)"
    )

    # --- Serialise model for multiprocessing ---
    model_params = {
        "alpha": model.alpha.tolist(),
        "beta": model.beta.tolist(),
        "rho": model.rho.tolist(),
        "thickness": model.thickness.tolist(),
        "Q_alpha": model.Q_alpha.tolist(),
        "Q_beta": model.Q_beta.tolist(),
    }

    # --- Parallelise over slowness batches ---
    if n_workers is None:
        n_workers = mp.cpu_count()
    n_workers = min(n_workers, np_slow)

    jp_all = list(range(np_slow))
    batch_size = max(1, np_slow // (n_workers * 4))  # ~4 batches per worker
    batches = [jp_all[i : i + batch_size] for i in range(0, np_slow, batch_size)]

    logger.info(f"Using {n_workers} workers, {len(batches)} batches")

    worker_fn = partial(
        _process_slowness_batch,
        p_samples=p_samples,
        omega_damped=omega_damped,
        offsets=offsets,
        dp=dp,
        model_params=model_params,
        free_surface=free_surface,
        taper=taper,
    )

    # --- Accumulate U(r, ω) = Σ_j contributions ---
    U = np.zeros((nr, nwm), dtype=np.complex128)

    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            for i_batch, batch_results in enumerate(
                pool.imap_unordered(worker_fn, batches)
            ):
                for _jp, contrib in batch_results:
                    U += contrib
                if (i_batch + 1) % max(1, len(batches) // 10) == 0:
                    pct = 100.0 * (i_batch + 1) / len(batches)
                    logger.info(f"  {pct:.0f}% complete")
    else:
        for i_batch, batch in enumerate(batches):
            batch_results = worker_fn(batch)
            for _jp, contrib in batch_results:
                U += contrib
            if (i_batch + 1) % max(1, len(batches) // 10) == 0:
                pct = 100.0 * (i_batch + 1) / len(batches)
                logger.info(f"  {pct:.0f}% complete")

    # Multiply by source spectrum
    U *= S[np.newaxis, :]

    logger.info("Slowness integration complete")

    # --- Inverse FFT to time domain with damping compensation ---
    gather = np.zeros((nr, nt), dtype=np.float64)
    for ir in range(nr):
        Uwk = np.zeros(nt, dtype=np.complex128)
        Uwk[1:nw] = U[ir, :]
        Uwk[nw + 1 :] = np.conj(U[ir, ::-1])

        seismogram_c = np.fft.fft(Uwk)
        gather[ir, :] = np.real(seismogram_c) * np.exp(-gamma * time)

    return time, offsets, gather


def plot_gather(
    time: np.ndarray,
    offsets: np.ndarray,
    gather: np.ndarray,
    output: str | Path,
    title: str = "Synthetic Seismogram Gather",
    t_max: float | None = None,
    clip: float = 0.8,
    norm: str = "global",
    noise_floor_db: float = -60.0,
) -> None:
    """Plot a seismogram gather with variable-area wiggle traces.

    Parameters
    ----------
    time : np.ndarray
        Time samples, shape (nt,).
    offsets : np.ndarray
        Offsets in km, shape (nr,).
    gather : np.ndarray
        Seismogram gather, shape (nr, nt).
    output : str or Path
        Output filename for the plot.
    title : str
        Plot title.
    t_max : float | None
        Maximum time to display (seconds). None shows all.
    clip : float
        Amplitude clip level as fraction of trace spacing (0-1).
    norm : str
        Normalisation mode: ``'global'`` (default) normalises all traces
        by the global peak amplitude — preserving true relative amplitudes
        and preventing integration noise at quiet traces from being
        amplified into visible artefacts. ``'trace'`` normalises each
        trace independently (useful for AVO analysis but prone to
        displaying numerical noise as spurious arrivals).
    noise_floor_db : float
        Noise floor in dB relative to the global peak (default: -60 dB).
        Only used when ``norm='global'``. Traces whose peak amplitude
        falls below this threshold are zeroed to suppress integration
        noise at far offsets / late times.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nr = len(offsets)
    dr = offsets[1] - offsets[0] if nr > 1 else 1.0

    if t_max is not None:
        tmask = time <= t_max
    else:
        tmask = np.ones(len(time), dtype=bool)
    t_plot = time[tmask]

    # Global peak amplitude for normalisation
    global_peak = np.max(np.abs(gather[:, tmask]))
    noise_threshold = global_peak * 10.0 ** (noise_floor_db / 20.0)

    fig, ax = plt.subplots(1, 1, figsize=(12, 14))

    for ir in range(nr):
        trace = gather[ir, tmask].copy()
        trace_peak = np.max(np.abs(trace))

        if norm == "global":
            # All traces share the same scale — true relative amplitudes
            if global_peak > 0:
                trace = trace / global_peak * dr * clip
            # Zero out traces below the noise floor
            if trace_peak < noise_threshold:
                trace[:] = 0.0
        else:
            # Per-trace normalisation (legacy behaviour)
            if trace_peak > 0:
                trace = trace / trace_peak * dr * clip

        x0 = offsets[ir]
        ax.plot(x0 + trace, t_plot, "k-", linewidth=0.4)
        ax.fill_betweenx(
            t_plot,
            x0,
            x0 + trace,
            where=(trace > 0),
            interpolate=True,
            color="black",
            alpha=0.9,
        )

    ax.set_xlabel("Offset (km)")
    ax.set_ylabel("Time (s)")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.set_xlim(offsets[0] - dr, offsets[-1] + dr)

    fig.tight_layout()
    plt.savefig(str(output), dpi=150, bbox_inches="tight")
    logger.info(f"Gather plot saved to {output}")
    plt.close()


def main() -> None:
    """CLI entry point for gather computation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute synthetic seismogram gather using discrete wavenumber summation.",
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
        default=20.0,
        help="Maximum offset in km (default: 20.0)",
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
        default=2048,
        dest="np_slow",
        help="Number of slowness samples (default: 2048)",
    )
    parser.add_argument(
        "--p-max",
        type=float,
        default=0.8,
        help="Maximum slowness in s/km (default: 0.8)",
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
        default=None,
        help="Maximum time to display in seconds (default: full window)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="seismogram_gather.png",
        help="Output plot filename (default: seismogram_gather.png)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plot generation, save data only",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: all CPU cores)",
    )
    parser.add_argument(
        "--free-surface",
        action="store_true",
        help="Include free surface reflections (water-column multiples)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    model = default_ocean_crust_model()
    offsets = np.arange(args.r_min, args.r_max + args.dr / 2, args.dr)

    logger.info(
        f"Model: {model.n_layers} layers, "
        f"{len(offsets)} offsets [{offsets[0]:.1f}-{offsets[-1]:.1f} km]"
    )

    time, offsets, gather = compute_gather(
        model,
        offsets,
        T=args.duration,
        nw=args.nw,
        np_slow=args.np_slow,
        p_max=args.p_max,
        gamma=args.gamma,
        n_workers=args.workers,
        free_surface=args.free_surface,
    )

    if not args.no_plot:
        try:
            fs_label = " + free surface" if args.free_surface else ""
            plot_gather(
                time,
                offsets,
                gather,
                output=args.output,
                title=(
                    f"Kennett Reflectivity Gather \u2014 5-layer ocean crust{fs_label}\n"
                    f"Discrete wavenumber summation (np={args.np_slow}, "
                    f"p_max={args.p_max} s/km)"
                ),
                t_max=args.t_max,
            )
        except ImportError:
            logger.warning("matplotlib not available, skipping plot")

    output_npz = Path(args.output).with_suffix(".npz")
    np.savez(
        output_npz,
        time=time,
        offsets=offsets,
        gather=gather,
    )
    logger.info(f"Data saved to {output_npz}")


if __name__ == "__main__":
    main()
