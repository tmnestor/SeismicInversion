"""Source wavelets for seismic modelling."""

from __future__ import annotations

import numpy as np

__all__ = ["ricker_spectrum", "ricker_wavelet"]


def ricker_spectrum(
    omega: np.ndarray,
    omega_max: float,
) -> np.ndarray:
    """
    Compute Ricker wavelet spectrum (frequency domain).

    Faithful translation of the S (source spectrum) calculation from kennetslo.f.

    Parameters
    ----------
    omega : np.ndarray
        Angular frequencies, shape (nfreq,).
    omega_max : float
        Maximum angular frequency used to define the wavelet.

    Returns
    -------
    np.ndarray
        Complex source spectrum, shape (nfreq,).
        S = Z * exp(-Z + i*omega*T0)
        where Z = omega²/(4α²), α = 0.1*omega_max, T0 = 5/(α*sqrt(2))

    Notes
    -----
    The Ricker wavelet in the frequency domain is a Gaussian envelope modulated
    by a quadratic phase term. The parameters α and T0 control the bandwidth
    and center time of the source.
    """
    alpha = 0.1 * omega_max
    T0 = 5.0 / (alpha * np.sqrt(2.0))

    Z = (omega**2) / (4.0 * alpha**2)
    S = Z * np.exp(-Z + 1j * omega * T0)

    return S


def ricker_wavelet(
    t: np.ndarray,
    f_peak: float,
) -> np.ndarray:
    """
    Compute Ricker wavelet in time domain.

    Parameters
    ----------
    t : np.ndarray
        Time samples, shape (ntime,).
    f_peak : float
        Peak frequency (Hz). The wavelet has maximum energy at this frequency.

    Returns
    -------
    np.ndarray
        Ricker wavelet samples, shape (ntime,).
        w(t) = (1 - 2π²f_peak²(t-t0)²) * exp(-π²f_peak²(t-t0)²)
        where t0 is chosen to center the wavelet around the peak.

    Notes
    -----
    The Ricker wavelet (also called Mexican Hat wavelet) is a zerophase wavelet
    commonly used in seismic applications. It has zero phase and a single main lobe.
    """
    omega_peak = 2 * np.pi * f_peak
    t0 = 1.0 / (np.pi * f_peak)  # Center the wavelet

    tau = np.pi * f_peak * (t - t0)
    w = (1.0 - 2.0 * tau**2) * np.exp(-(tau**2))

    return w
