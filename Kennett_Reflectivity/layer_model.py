"""Layer model for plane-stratified elastic media.

Faithful translation of CMPLXSLO and VERTICALSLO from kennetslo.f.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["LayerModel", "complex_slowness", "vertical_slowness"]


def complex_slowness(velocity: float, Q: float) -> complex:
    """
    Compute complex slowness accounting for attenuation.

    Implements CMPLXSLO from kennetslo.f:
        twoQ = 2*Q
        twoQsq = twoQ*twoQ
        twoQsqp1byC = (1 + twoQsq)*C
        cmplxslo = cmplx(twoQsq/twoQsqp1byC, twoQ/twoQsqp1byC)

    Parameters
    ----------
    velocity : float
        Phase velocity C.
    Q : float
        Quality factor.

    Returns
    -------
    complex
        Complex slowness s = Re(s) + i*Im(s).
    """
    if velocity <= 0:
        raise ValueError(f"velocity must be positive, got {velocity}")
    if Q <= 0:
        raise ValueError(f"Q must be positive, got {Q}")

    twoQ = 2.0 * Q
    twoQsq = twoQ * twoQ
    twoQsqp1byC = (1.0 + twoQsq) * velocity
    return complex(twoQsq / twoQsqp1byC, twoQ / twoQsqp1byC)


def vertical_slowness(slowness: complex, p: complex) -> complex:
    """
    Compute vertical slowness with Im(eta) > 0 branch cut convention.

    Implements VERTICALSLO from kennetslo.f:
        T = (C+P)*(C-P)
        VERTICALSLO = csqrt(T)
        IF(AIMAG(VERTICALSLO) .LE. 0.0) VERTICALSLO = -VERTICALSLO

    Parameters
    ----------
    slowness : complex
        Complex slowness of the layer.
    p : complex or float
        Horizontal slowness (ray parameter).

    Returns
    -------
    complex
        Vertical slowness eta with Im(eta) > 0.
    """
    T = (slowness + p) * (slowness - p)
    eta = np.sqrt(T)
    # Fortran uses .LE. 0.0 (less than or equal)
    if eta.imag <= 0.0:
        eta = -eta
    return eta


@dataclass
class LayerModel:
    """
    Model of a plane-stratified elastic half-space.

    Attributes
    ----------
    alpha : np.ndarray
        P-wave velocities (real), shape (n_layers,).
    beta : np.ndarray
        S-wave velocities (real, 0 for acoustic), shape (n_layers,).
    rho : np.ndarray
        Mass densities, shape (n_layers,).
    thickness : np.ndarray
        Layer thicknesses (np.inf for half-space), shape (n_layers,).
    Q_alpha : np.ndarray
        P-wave quality factors, shape (n_layers,).
    Q_beta : np.ndarray
        S-wave quality factors, shape (n_layers,).
    """

    alpha: np.ndarray
    beta: np.ndarray
    rho: np.ndarray
    thickness: np.ndarray
    Q_alpha: np.ndarray
    Q_beta: np.ndarray

    def __post_init__(self) -> None:
        self.alpha = np.asarray(self.alpha, dtype=np.float64)
        self.beta = np.asarray(self.beta, dtype=np.float64)
        self.rho = np.asarray(self.rho, dtype=np.float64)
        self.thickness = np.asarray(self.thickness, dtype=np.float64)
        self.Q_alpha = np.asarray(self.Q_alpha, dtype=np.float64)
        self.Q_beta = np.asarray(self.Q_beta, dtype=np.float64)

        n = self.n_layers
        for name in ("alpha", "beta", "rho", "thickness", "Q_alpha", "Q_beta"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} length mismatch: expected {n}")

        if np.any(self.alpha <= 0):
            raise ValueError("All alpha must be positive")
        if np.any(self.beta < 0):
            raise ValueError("All beta must be non-negative")
        if np.any(self.rho <= 0):
            raise ValueError("All rho must be positive")

    @property
    def n_layers(self) -> int:
        return len(self.alpha)

    @classmethod
    def from_arrays(cls, alpha, beta, rho, thickness, Q_alpha, Q_beta) -> LayerModel:
        return cls(
            alpha=np.asarray(alpha, dtype=np.float64),
            beta=np.asarray(beta, dtype=np.float64),
            rho=np.asarray(rho, dtype=np.float64),
            thickness=np.asarray(thickness, dtype=np.float64),
            Q_alpha=np.asarray(Q_alpha, dtype=np.float64),
            Q_beta=np.asarray(Q_beta, dtype=np.float64),
        )

    def complex_slowness_p(self) -> np.ndarray:
        """Complex P-wave slowness for each layer. Matches Fortran A(IL)."""
        return np.array(
            [
                complex_slowness(self.alpha[i], self.Q_alpha[i])
                for i in range(self.n_layers)
            ],
            dtype=np.complex128,
        )

    def complex_slowness_s(self) -> np.ndarray:
        """Complex S-wave slowness for each layer. Matches Fortran B(IL)."""
        result = np.zeros(self.n_layers, dtype=np.complex128)
        for i in range(self.n_layers):
            if self.beta[i] > 0:
                result[i] = complex_slowness(self.beta[i], self.Q_beta[i])
            # else: 0+0j (acoustic layer, Fortran B(1)=C0)
        return result

    def complex_velocity_s(self) -> np.ndarray:
        """Complex S-wave velocity for each layer. Matches Fortran BETA(IL) = C1/B(IL)."""
        s_s = self.complex_slowness_s()
        result = np.zeros(self.n_layers, dtype=np.complex128)
        for i in range(self.n_layers):
            if abs(s_s[i]) > 0:
                result[i] = 1.0 / s_s[i]
            # else: 0+0j (acoustic, Fortran BETA(1)=C0)
        return result
