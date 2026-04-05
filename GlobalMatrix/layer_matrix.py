"""Displacement-stress eigenvector matrices for the Global Matrix Method.

Constructs the E-matrices that relate wave amplitudes (downgoing/upgoing P and S)
to displacement-stress vectors [u_x, u_z, sigma_zz/(-iw), sigma_xz/(-iw)] at
layer boundaries.

Convention: exp(-iwt) inverse Fourier transform, depth positive downward.
"""

import numpy as np
import torch

__all__ = [
    "layer_eigenvectors",
    "layer_eigenvectors_batched",
    "layer_eigenvectors_sh_batched",
    "layer_eigenvectors_torch",
    "ocean_eigenvectors",
    "ocean_eigenvectors_batched",
    "ocean_eigenvectors_torch",
]


def layer_eigenvectors(
    p: complex,
    eta: complex,
    neta: complex,
    rho: float,
    beta_c: complex,
) -> tuple[np.ndarray, np.ndarray]:
    """Build 4x2 displacement-stress eigenvector matrices for a solid layer.

    Args:
        p: Horizontal slowness (ray parameter).
        eta: Vertical P-wave slowness.
        neta: Vertical S-wave slowness.
        rho: Layer density.
        beta_c: Complex S-wave velocity (1/complex_s_slowness).

    Returns:
        (E_d, E_u) each shape (4, 2). Columns are [P, S].
        Rows: [u_x, u_z, sigma_zz/(-iw), sigma_xz/(-iw)].
    """
    mu = rho * beta_c**2
    gamma = rho * (1.0 - 2.0 * beta_c**2 * p**2)

    E_d = np.array(
        [
            [p, neta],
            [eta, -p],
            [-gamma, 2.0 * mu * p * neta],
            [2.0 * mu * p * eta, gamma],
        ],
        dtype=np.complex128,
    )

    E_u = np.array(
        [
            [p, -neta],
            [-eta, -p],
            [-gamma, -2.0 * mu * p * neta],
            [-2.0 * mu * p * eta, gamma],
        ],
        dtype=np.complex128,
    )

    return E_d, E_u


def ocean_eigenvectors(
    p: complex,
    eta: complex,
    rho: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build eigenvector columns for an acoustic (ocean) layer.

    For the acoustic layer there is only a P-wave mode. The displacement-stress
    vector has two components: [u_z, sigma_zz/(-iw)].

    Args:
        p: Horizontal slowness.
        eta: Vertical P-wave slowness in the ocean.
        rho: Ocean density.

    Returns:
        (e_d, e_u) each shape (2,) — downgoing and upgoing P-wave columns.
        Row 0: u_z, Row 1: sigma_zz/(-iw) = rho (for unit amplitude).
    """
    e_d = np.array([eta, -rho], dtype=np.complex128)
    e_u = np.array([-eta, -rho], dtype=np.complex128)
    return e_d, e_u


# ---- Batched (kH-vectorised) versions ----


def layer_eigenvectors_batched(
    p: np.ndarray,
    eta: np.ndarray,
    neta: np.ndarray,
    rho: float,
    beta_c: complex,
) -> tuple[np.ndarray, np.ndarray]:
    """Build 4x2 eigenvector matrices for a solid layer, batched over kH.

    Args:
        p: Horizontal slowness, shape (n_kH,).
        eta: Vertical P-wave slowness, shape (n_kH,).
        neta: Vertical S-wave slowness, shape (n_kH,).
        rho: Layer density (scalar).
        beta_c: Complex S-wave velocity (scalar).

    Returns:
        (E_d, E_u) each shape (n_kH, 4, 2). Columns are [P, S].
        Rows: [u_x, u_z, sigma_zz/(-iw), sigma_xz/(-iw)].
    """
    mu = rho * beta_c**2
    gamma = rho * (1.0 - 2.0 * beta_c**2 * p**2)
    two_mu_p = 2.0 * mu * p

    # E_d: shape (n_kH, 4, 2)
    E_d = np.stack(
        [
            np.stack([p, neta], axis=-1),
            np.stack([eta, -p], axis=-1),
            np.stack([-gamma, two_mu_p * neta], axis=-1),
            np.stack([two_mu_p * eta, gamma], axis=-1),
        ],
        axis=1,
    ).astype(np.complex128)

    # E_u: sign-flip upgoing columns
    E_u = np.stack(
        [
            np.stack([p, -neta], axis=-1),
            np.stack([-eta, -p], axis=-1),
            np.stack([-gamma, -two_mu_p * neta], axis=-1),
            np.stack([-two_mu_p * eta, gamma], axis=-1),
        ],
        axis=1,
    ).astype(np.complex128)

    return E_d, E_u


def ocean_eigenvectors_batched(
    p: np.ndarray,
    eta: np.ndarray,
    rho: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build eigenvector columns for an acoustic layer, batched over kH.

    Args:
        p: Horizontal slowness, shape (n_kH,).
        eta: Vertical P-wave slowness in the ocean, shape (n_kH,).
        rho: Ocean density (scalar).

    Returns:
        (e_d, e_u) each shape (n_kH, 2).
        Row 0: u_z, Row 1: sigma_zz/(-iw).
    """
    rho_arr = np.full_like(eta, -rho)
    e_d = np.stack([eta, rho_arr], axis=-1).astype(np.complex128)
    e_u = np.stack([-eta, rho_arr], axis=-1).astype(np.complex128)
    return e_d, e_u


def layer_eigenvectors_sh_batched(
    neta: np.ndarray,
    rho: float,
    beta_c: complex,
) -> tuple[np.ndarray, np.ndarray]:
    """SH eigenvector columns for a solid layer, batched over kH.

    The SH system has 2 components (u_t, sigma_tz/(-iw)) and 1 wave type
    per direction. Downgoing: e_d = [[-neta], [mu*neta]],
    upgoing: e_u = [[neta], [mu*neta]].

    Args:
        neta: Vertical S-wave slowness, shape (n_kH,).
        rho: Layer density (scalar).
        beta_c: Complex S-wave velocity (scalar).

    Returns:
        (e_d_sh, e_u_sh) each shape (n_kH, 2, 1).
    """
    mu = rho * beta_c**2
    mu_neta = mu * neta
    # e_d: [[-neta], [mu*neta]]
    e_d = np.stack([-neta, mu_neta], axis=-1).astype(np.complex128)[:, :, np.newaxis]
    # e_u: [[neta], [mu*neta]]
    e_u = np.stack([neta, mu_neta], axis=-1).astype(np.complex128)[:, :, np.newaxis]
    return e_d, e_u


# ---- PyTorch versions ----

_CDTYPE = torch.complex128


def layer_eigenvectors_torch(
    p: torch.Tensor,
    eta: torch.Tensor,
    neta: torch.Tensor,
    rho: torch.Tensor,
    beta_c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build 4x2 eigenvector matrices for a solid layer (PyTorch).

    Args:
        p: Horizontal slowness (complex128 scalar tensor).
        eta: Vertical P-wave slowness (complex128 scalar tensor).
        neta: Vertical S-wave slowness (complex128 scalar tensor).
        rho: Layer density (float64 scalar tensor).
        beta_c: Complex S-wave velocity (complex128 scalar tensor).

    Returns:
        (E_d, E_u) each shape (4, 2), dtype complex128.
    """
    cp = p.to(_CDTYPE)
    crho = rho.to(_CDTYPE)
    mu = crho * beta_c * beta_c
    gamma = crho * (1.0 - 2.0 * beta_c * beta_c * cp * cp)

    E_d = torch.stack(
        [
            torch.stack([cp, neta]),
            torch.stack([eta, -cp]),
            torch.stack([-gamma, 2.0 * mu * cp * neta]),
            torch.stack([2.0 * mu * cp * eta, gamma]),
        ]
    )

    E_u = torch.stack(
        [
            torch.stack([cp, -neta]),
            torch.stack([-eta, -cp]),
            torch.stack([-gamma, -2.0 * mu * cp * neta]),
            torch.stack([-2.0 * mu * cp * eta, gamma]),
        ]
    )

    return E_d, E_u


def ocean_eigenvectors_torch(
    p: torch.Tensor,
    eta: torch.Tensor,
    rho: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build eigenvector columns for an acoustic layer (PyTorch).

    Args:
        p: Horizontal slowness (complex128 scalar tensor).
        eta: Vertical P-wave slowness (complex128 scalar tensor).
        rho: Ocean density (float64 scalar tensor).

    Returns:
        (e_d, e_u) each shape (2,), dtype complex128.
    """
    crho = rho.to(_CDTYPE)
    e_d = torch.stack([eta, -crho])
    e_u = torch.stack([-eta, -crho])
    return e_d, e_u
