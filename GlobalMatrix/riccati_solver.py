"""Block-Riccati sweep solver for the Global Matrix Method.

Propagates a 2x2 Riccati reflection matrix upward through the layer stack,
solving only 4x4 block systems at each interface. This reduces the per-frequency
cost from O(N^3) (dense solve) to O(N) in the number of layers.

Reference: Nestor (1991) thesis — optimised Block Riccati solver for the
block-tridiagonal global matrix system.
"""

import numpy as np
import torch


__all__ = [
    "riccati_sweep_numpy",
    "riccati_sweep_torch",
]

_CDTYPE_NP = np.complex128
_CDTYPE_TORCH = torch.complex128


def riccati_sweep_numpy(
    nlayer: int,
    E_d: dict[int, np.ndarray],
    E_u: dict[int, np.ndarray],
    phase_d: dict[int, np.ndarray],
    e_d_oc: np.ndarray,
    e_u_oc: np.ndarray,
    e0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Block-Riccati sweep for reflectivity (NumPy).

    Args:
        nlayer: Total number of layers (ocean + elastic + half-space).
        E_d: Downgoing eigenvector matrices, E_d[j] shape (4, 2) for j=1..nlayer-1.
        E_u: Upgoing eigenvector matrices, E_u[j] shape (4, 2) for j=1..nlayer-1.
        phase_d: Phase diagonals, phase_d[j] shape (nfreq, 2) for j=1..nlayer-2.
        e_d_oc: Ocean downgoing eigenvector, shape (2,).
        e_u_oc: Ocean upgoing eigenvector, shape (2,).
        e0: Ocean phase exp(i*w*eta0*h0), shape (nfreq,).

    Returns:
        (R, U0_P): Reflectivity and upgoing ocean amplitude, each shape (nfreq,).
    """
    nfreq = e0.shape[0]
    M = nlayer - 2  # number of finite elastic layers

    if M == 0:
        # 2-layer model: ocean + half-space, no finite elastic layers
        return _two_layer_numpy(E_d, e_d_oc, e_u_oc, e0, nfreq)

    # --- Upward Riccati sweep (bottom → top) ---
    # Initialise Y = 0 (radiation condition at half-space)
    Y = np.zeros((nfreq, 2, 2), dtype=_CDTYPE_NP)

    # k goes from M down to 1 (layer indices of finite elastic layers)
    for k in range(M, 0, -1):
        k_below = k + 1  # layer below interface k

        # Build E_eff = E_d[k_below] + E_u[k_below] * diag(phase[k_below]) * Y
        if k_below < nlayer - 1:
            # k_below is a finite elastic layer
            # phase_d[k_below] is (nfreq, 2)
            # E_u[k_below] * diag(phase) -> (nfreq, 4, 2)
            E_u_phased = (
                E_u[k_below][np.newaxis, :, :] * phase_d[k_below][:, np.newaxis, :]
            )
            # E_eff = E_d[k_below] + E_u_phased @ Y, shape (nfreq, 4, 2)
            E_eff = E_d[k_below][np.newaxis, :, :] + E_u_phased @ Y
        else:
            # k_below is the half-space: no upgoing, E_eff = E_d[k_below]
            E_eff = np.broadcast_to(
                E_d[k_below][np.newaxis, :, :], (nfreq, 4, 2)
            ).copy()

        # Build A_d = E_d[k] * diag(phase[k]), shape (nfreq, 4, 2)
        A_d = E_d[k][np.newaxis, :, :] * phase_d[k][:, np.newaxis, :]

        # Build F = [E_u[k], -E_eff], shape (nfreq, 4, 4)
        E_u_k = np.broadcast_to(E_u[k][np.newaxis, :, :], (nfreq, 4, 2))
        F = np.concatenate([E_u_k, -E_eff], axis=-1)  # (nfreq, 4, 4)

        # Solve F @ X = -A_d for X, shape (nfreq, 4, 2)
        X = np.linalg.solve(F, -A_d)

        # Extract Y = top 2 rows of X
        Y = X[:, :2, :]

    # --- Ocean-bottom extraction ---
    # Y is now Y_1, the Riccati matrix at the top of layer 1
    # Build E1_eff = E_d[1] + E_u[1] * diag(phase[1]) * Y_1
    E_u_phased_1 = E_u[1][np.newaxis, :, :] * phase_d[1][:, np.newaxis, :]
    E1_eff = E_d[1][np.newaxis, :, :] + E_u_phased_1 @ Y  # (nfreq, 4, 2)

    return _ocean_extraction_numpy(E1_eff, e_d_oc, e_u_oc, e0, nfreq)


def _two_layer_numpy(
    E_d: dict[int, np.ndarray],
    e_d_oc: np.ndarray,
    e_u_oc: np.ndarray,
    e0: np.ndarray,
    nfreq: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Handle 2-layer model (ocean + half-space) directly."""
    # E1_eff = E_d[1] (half-space, no upgoing)
    E1_eff = np.broadcast_to(E_d[1][np.newaxis, :, :], (nfreq, 4, 2)).copy()
    return _ocean_extraction_numpy(E1_eff, e_d_oc, e_u_oc, e0, nfreq)


def _ocean_extraction_numpy(
    E1_eff: np.ndarray,
    e_d_oc: np.ndarray,
    e_u_oc: np.ndarray,
    e0: np.ndarray,
    nfreq: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract reflectivity from E1_eff at the ocean bottom.

    Continuity conditions at ocean bottom:
      u_z:       e_u_oc[0]*U0P + e_d_oc[0]*e0 = E1_eff[row1,:] @ D1
      sigma_zz:  e_u_oc[1]*U0P + e_d_oc[1]*e0 = E1_eff[row2,:] @ D1
      sigma_xz:  0 = E1_eff[row3,:] @ D1

    E-matrix rows: 0=u_x, 1=u_z, 2=sigma_zz, 3=sigma_xz

    Form a 2x2 system for D1 by eliminating U0P:
      Row A: e_u_oc[0]*E1_eff[:,2,:] - e_u_oc[1]*E1_eff[:,1,:]  (eliminates U0P)
      Row B: E1_eff[:,3,:]  (sigma_xz = 0, no U0P involved)
    RHS: [(e_u_oc[0]*e_d_oc[1] - e_u_oc[1]*e_d_oc[0])*e0, 0]
    """
    # Build 2x2 system, shape (nfreq, 2, 2)
    row_A = e_u_oc[0] * E1_eff[:, 2, :] - e_u_oc[1] * E1_eff[:, 1, :]  # (nfreq, 2)
    row_B = E1_eff[:, 3, :]  # (nfreq, 2)
    mat = np.stack([row_A, row_B], axis=1)  # (nfreq, 2, 2)

    # RHS, shape (nfreq, 2)
    rhs_A = (e_u_oc[0] * e_d_oc[1] - e_u_oc[1] * e_d_oc[0]) * e0  # (nfreq,)
    rhs = np.stack([rhs_A, np.zeros(nfreq, dtype=_CDTYPE_NP)], axis=1)  # (nfreq, 2)

    # Solve for D1, shape (nfreq, 2)
    # Use 3D rhs to avoid numpy batched solve ambiguity
    D1 = np.linalg.solve(mat, rhs[..., np.newaxis]).squeeze(-1)

    # Recover U0P from u_z continuity:
    # e_u_oc[0]*U0P = E1_eff[:,1,:] @ D1 - e_d_oc[0]*e0
    U0_P = (np.einsum("fi,fi->f", E1_eff[:, 1, :], D1) - e_d_oc[0] * e0) / e_u_oc[0]

    # Reflectivity R = e0 * U0_P
    R = e0 * U0_P

    return R, U0_P


# ---- PyTorch version ----


def riccati_sweep_torch(
    nlayer: int,
    E_d: dict[int, torch.Tensor],
    E_u: dict[int, torch.Tensor],
    phase_d: dict[int, torch.Tensor],
    e_d_oc: torch.Tensor,
    e_u_oc: torch.Tensor,
    e0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-Riccati sweep for reflectivity (PyTorch, differentiable).

    Args:
        nlayer: Total number of layers.
        E_d: Downgoing eigenvector matrices, E_d[j] shape (4, 2) for j=1..nlayer-1.
        E_u: Upgoing eigenvector matrices, E_u[j] shape (4, 2) for j=1..nlayer-1.
        phase_d: Phase diagonals, phase_d[j] shape (nfreq, 2) for j=1..nlayer-2.
        e_d_oc: Ocean downgoing eigenvector, shape (2,).
        e_u_oc: Ocean upgoing eigenvector, shape (2,).
        e0: Ocean phase, shape (nfreq,).

    Returns:
        (R, U0_P): Reflectivity and upgoing ocean amplitude, each shape (nfreq,).
    """
    nfreq = e0.shape[0]
    M = nlayer - 2

    if M == 0:
        return _two_layer_torch(E_d, e_d_oc, e_u_oc, e0, nfreq)

    # Initialise Y = 0
    Y = torch.zeros((nfreq, 2, 2), dtype=_CDTYPE_TORCH)

    for k in range(M, 0, -1):
        k_below = k + 1

        if k_below < nlayer - 1:
            # Finite elastic layer below
            E_u_phased = E_u[k_below].unsqueeze(0) * phase_d[k_below].unsqueeze(1)
            E_eff = E_d[k_below].unsqueeze(0) + E_u_phased @ Y
        else:
            # Half-space below
            E_eff = E_d[k_below].unsqueeze(0).expand(nfreq, 4, 2)

        # A_d = E_d[k] * diag(phase[k])
        A_d = E_d[k].unsqueeze(0) * phase_d[k].unsqueeze(1)

        # F = [E_u[k], -E_eff]
        E_u_k = E_u[k].unsqueeze(0).expand(nfreq, 4, 2)
        F = torch.cat([E_u_k, -E_eff], dim=-1)  # (nfreq, 4, 4)

        # Solve F @ X = -A_d
        X = torch.linalg.solve(F, -A_d)

        Y = X[:, :2, :]

    # Ocean extraction
    E_u_phased_1 = E_u[1].unsqueeze(0) * phase_d[1].unsqueeze(1)
    E1_eff = E_d[1].unsqueeze(0) + E_u_phased_1 @ Y

    return _ocean_extraction_torch(E1_eff, e_d_oc, e_u_oc, e0, nfreq)


def _two_layer_torch(
    E_d: dict[int, torch.Tensor],
    e_d_oc: torch.Tensor,
    e_u_oc: torch.Tensor,
    e0: torch.Tensor,
    nfreq: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Handle 2-layer model (ocean + half-space) in PyTorch."""
    E1_eff = E_d[1].unsqueeze(0).expand(nfreq, 4, 2)
    return _ocean_extraction_torch(E1_eff, e_d_oc, e_u_oc, e0, nfreq)


def _ocean_extraction_torch(
    E1_eff: torch.Tensor,
    e_d_oc: torch.Tensor,
    e_u_oc: torch.Tensor,
    e0: torch.Tensor,
    nfreq: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract reflectivity from E1_eff at ocean bottom (PyTorch)."""
    row_A = e_u_oc[0] * E1_eff[:, 2, :] - e_u_oc[1] * E1_eff[:, 1, :]
    row_B = E1_eff[:, 3, :]
    mat = torch.stack([row_A, row_B], dim=1)

    rhs_A = (e_u_oc[0] * e_d_oc[1] - e_u_oc[1] * e_d_oc[0]) * e0
    rhs = torch.stack([rhs_A, torch.zeros(nfreq, dtype=_CDTYPE_TORCH)], dim=1)

    D1 = torch.linalg.solve(mat, rhs.unsqueeze(-1)).squeeze(-1)

    U0_P = (torch.sum(E1_eff[:, 1, :] * D1, dim=-1) - e_d_oc[0] * e0) / e_u_oc[0]
    R = e0 * U0_P

    return R, U0_P
