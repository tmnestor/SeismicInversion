"""Compute synthetic seismograms using Kennett reflectivity method.

Faithful translation of the main KennetSlo program from kennetslo.f.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import collections.abc

import numpy as np

from .kennett_reflectivity import kennett_reflectivity
from .layer_model import LayerModel
from .source import ricker_spectrum

__all__ = ["compute_seismogram", "default_ocean_crust_model"]

logger = logging.getLogger(__name__)


def default_ocean_crust_model() -> LayerModel:
    """
    Create the default 5-layer ocean-crust model from kennetslo.f.

    Returns
    -------
    LayerModel
        Layer 1 (ocean):        alpha=1.5, beta=0.0, rho=1.0, h=2.0, Qa=20000, Qb=1e10
        Layer 2 (sediment):     alpha=1.6, beta=0.3, rho=2.0, h=1.0, Qa=100, Qb=100
        Layer 3 (crust):        alpha=3.0, beta=1.5, rho=3.0, h=1.0, Qa=100, Qb=100
        Layer 4 (upper mantle): alpha=5.0, beta=3.0, rho=3.0, h=1.0, Qa=100, Qb=100
        Layer 5 (half-space):   alpha=2.2, beta=1.1, rho=1.8, h=inf, Qa=100, Qb=100

    Notes
    -----
    Fortran uses NLP1=4 meaning 4 interfaces (5 layers), with H values
    as cumulative depths. DD(il) = H(il) - H(il-1) gives thicknesses:
        DD(1) = H(1) - 0.5*(Ds+Dr) = 2.0 (ocean thickness)
        DD(2) = H(2) - H(1) = 1.0 (sediment)
        DD(3) = H(3) - H(2) = 1.0 (crust)
        DD(4) = H(4) - H(3) = 1.0 (upper mantle)
    """
    return LayerModel.from_arrays(
        alpha=[1.5, 1.6, 3.0, 5.0, 2.2],
        beta=[0.0, 0.3, 1.5, 3.0, 1.1],
        rho=[1.0, 2.0, 3.0, 3.0, 1.8],
        thickness=[2.0, 1.0, 1.0, 1.0, np.inf],
        Q_alpha=[20000, 100, 100, 100, 100],
        Q_beta=[1e10, 100, 100, 100, 100],
    )


def compute_seismogram(
    model: LayerModel,
    p: float,
    T: float = 64.0,
    nw: int = 2048,
    eps: float = 0.0,
    source_func: collections.abc.Callable | None = None,
    free_surface: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute synthetic seismogram using Kennett's reflectivity method.

    Faithful translation of the KennetSlo main program orchestration.

    Parameters
    ----------
    model : LayerModel
        The stratified elastic model.
    p : float
        Horizontal slowness (ray parameter).
    T : float, optional
        Time window (seconds), default 64.0.
    nw : int, optional
        Number of positive frequencies (must be power of 2), default 2048.
    eps : float, optional
        Frequency damping factor, default 0.0.
    source_func : callable, optional
        Source spectrum function(omega, omega_max) -> S. Default: Ricker.

    Returns
    -------
    time : np.ndarray
        Time samples, shape (2*nw,).
    seismogram : np.ndarray
        Real-valued seismogram, shape (2*nw,).

    Notes
    -----
    Matches the Fortran frequency grid and IFFT convention:
        - Frequency samples: omega = [dw, 2*dw, ..., (nw-1)*dw]
        - Hermitian spectrum of length NT=2*NW for real-valued output
        - Time step: dt = T/(2*NW)
    """
    logger.info(
        f"Computing seismogram: {model.n_layers} layers, p={p:.6f}, T={T:.1f}s, nw={nw}"
    )

    # Frequency grid (Fortran: DW = 2*PI/T, frequencies from DW to NWM*DW)
    dw = 2.0 * np.pi / T
    nwm = nw - 1  # Fortran NWM = NW - 1
    wmax = nw * dw  # Fortran WMAX = NW*DW

    # Frequency samples: iw=1 to nwm, omega = iw*dw
    omega = np.arange(1, nwm + 1, dtype=np.float64) * dw  # shape (nwm,)

    # Compute source spectrum (Fortran: M(iw) = S(w, wmax))
    if source_func is None:
        S = ricker_spectrum(omega, wmax)
    else:
        S = source_func(omega, wmax)
    logger.debug(f"Source spectrum computed: {nwm} frequencies")

    # Compute reflectivity (Fortran: SLOWRESP + Kennet_Reflex)
    R = kennett_reflectivity(model, p, omega, free_surface=free_surface)
    logger.debug(f"Reflectivity computed: max|R| = {np.max(np.abs(R)):.6f}")

    # Multiply reflectivity by source spectrum (Fortran: U(iw,ip) = U(iw,ip)*M(iw))
    Y = R * S  # shape (nwm,)

    # Build Hermitian spectrum for real-valued IFFT (Fortran convention)
    # NT = 2*NW, Uwk(1)=0, Uwk(2:NW)=U(1:NWM), Uwk(NW+1)=0,
    # Uwk(NW+2:NT)=conj(U(NWM:1))
    nt = 2 * nw
    Uwk = np.zeros(nt, dtype=np.complex128)
    Uwk[1:nw] = Y  # positive frequencies (indices 1 to NW-1)
    # Uwk[nw] = 0            # Nyquist (already zero)
    Uwk[nw + 1 :] = np.conj(Y[::-1])  # negative frequencies

    # Fortran: CALL FFT(Uwk, NT, -1.0)
    # The Fortran FFT with SIGNI=-1 computes: Y(n) = sum_k X(k)*exp(-i*2*pi*n*k/N)
    # This is numpy.fft.fft (forward DFT), NOT ifft.
    # With Hermitian input X(k) = conj(X(N-k)), the output is real-valued.
    seismogram_c = np.fft.fft(Uwk)

    # Apply frequency damping compensation (Fortran: ff=Real(Uwk(it))*exp(eps*TT))
    dt = T / float(nt)
    time = np.arange(nt, dtype=np.float64) * dt
    seismogram = np.real(seismogram_c) * np.exp(eps * time)

    logger.info(
        f"Seismogram computed: dt={dt:.6f}s, {nt} samples, duration={time[-1]:.1f}s"
    )

    return time, seismogram


def main() -> None:
    """CLI entry point for seismogram computation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute synthetic seismogram using Kennett reflectivity.",
    )
    parser.add_argument(
        "-p",
        "--slowness",
        type=float,
        default=0.2,
        help="Horizontal slowness / ray parameter in s/km (default: 0.2)",
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
        "-o",
        "--output",
        type=str,
        default="seismogram_verification.png",
        help="Output plot filename (default: seismogram_verification.png)",
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
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    model = default_ocean_crust_model()
    logger.info(f"Model: {model.n_layers} layers")
    logger.info(f"Computing seismogram for p={args.slowness:.6f}...")

    time, seismogram = compute_seismogram(
        model,
        args.slowness,
        T=args.duration,
        nw=args.nw,
        free_surface=args.free_surface,
    )

    if not args.no_plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

            # --- Top panel: full seismogram (variable-area wiggle trace) ---
            ax1.plot(time, seismogram, "k-", linewidth=0.5)
            ax1.fill_between(
                time,
                0,
                seismogram,
                where=(seismogram > 0),
                interpolate=True,
                color="black",
                alpha=0.9,
            )
            ax1.axhline(0, color="k", linewidth=0.3)
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Amplitude")
            ax1.set_title(
                f"Kennett Reflectivity \u2014 5-layer ocean crust"
                f" (p={args.slowness} s/km)"
            )

            # --- Bottom panel: early arrivals detail (0-15s) ---
            mask = time <= 15.0
            ax2.plot(time[mask], seismogram[mask], "k-", linewidth=0.5)
            ax2.fill_between(
                time[mask],
                0,
                seismogram[mask],
                where=(seismogram[mask] > 0),
                interpolate=True,
                color="black",
                alpha=0.9,
            )
            ax2.axhline(0, color="k", linewidth=0.3)

            # Mark expected ocean-layer two-way time
            t_ocean = 2.0 * model.thickness[0] / model.alpha[0]
            ax2.axvline(
                t_ocean,
                color="r",
                linestyle="--",
                alpha=0.6,
                label=(r"$2h_{\mathrm{ocean}}/\alpha_{\mathrm{ocean}}$"),
            )
            ax2.legend(fontsize=10)
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Amplitude")
            ax2.set_title("Early arrivals detail (0-15s)")

            fig.tight_layout()
            plt.savefig(args.output, dpi=150, bbox_inches="tight")
            logger.info(f"Plot saved to {args.output}")
            plt.close()
        except ImportError:
            logger.warning("matplotlib not available, skipping plot")

        # --- ObsPy Stream export (for interoperability with seismic formats) ---
        # try:
        #     import obspy
        #     dt = time[1] - time[0]
        #     tr = obspy.Trace(data=seismogram.copy())
        #     tr.stats.delta = dt
        #     tr.stats.network = "SY"
        #     tr.stats.station = "KNTT"
        #     tr.stats.channel = "BHZ"
        #     st = obspy.Stream([tr])
        #     st.write(args.output.replace(".png", ".mseed"), format="MSEED")
        # except ImportError:
        #     logger.warning("obspy not available, skipping MSEED export")

    output_data = f"seismogram_p{args.slowness:.4f}.txt"
    np.savetxt(output_data, np.column_stack([time, seismogram]), fmt="%.8e")
    logger.info(f"Data saved to {output_data}")


if __name__ == "__main__":
    main()
