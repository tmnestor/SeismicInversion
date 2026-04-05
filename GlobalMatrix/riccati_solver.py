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
    "compute_source_vector",
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
    source_terms: dict[int, np.ndarray] | None = None,
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
        source_terms: Optional dict mapping interface index k to source vector
            σ_k of shape (nfreq, 4). Key 0 = ocean bottom, keys 1..M =
            solid-solid interfaces. Produced by ``compute_source_vector``.

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

    if source_terms is None:
        source_terms = {}
    has_sources = len(source_terms) > 0
    if has_sources:
        g = np.zeros((nfreq, 2), dtype=_CDTYPE_NP)

    # k goes from M down to 1 (layer indices of finite elastic layers)
    for k in range(M, 0, -1):
        k_below = k + 1  # layer below interface k

        # Build E_eff = E_d[k_below] + E_u[k_below] * diag(phase[k_below]) * Y
        if k_below < nlayer - 1:
            # k_below is a finite elastic layer
            E_u_phased = (
                E_u[k_below][np.newaxis, :, :] * phase_d[k_below][:, np.newaxis, :]
            )
            E_eff = E_d[k_below][np.newaxis, :, :] + E_u_phased @ Y
        else:
            # k_below is the half-space: no upgoing, E_eff = E_d[k_below]
            E_u_phased = None
            E_eff = np.broadcast_to(
                E_d[k_below][np.newaxis, :, :], (nfreq, 4, 2)
            ).copy()

        # Build A_d = E_d[k] * diag(phase[k]), shape (nfreq, 4, 2)
        A_d = E_d[k][np.newaxis, :, :] * phase_d[k][:, np.newaxis, :]

        # Build F = [E_u[k], -E_eff], shape (nfreq, 4, 4)
        E_u_k = np.broadcast_to(E_u[k][np.newaxis, :, :], (nfreq, 4, 2))
        F = np.concatenate([E_u_k, -E_eff], axis=-1)  # (nfreq, 4, 4)

        if has_sources:
            # Build particular-solution RHS: σ_k + E_u_phased @ g
            rhs_p = np.zeros((nfreq, 4), dtype=_CDTYPE_NP)
            if E_u_phased is not None:
                rhs_p = rhs_p + np.einsum("fij,fj->fi", E_u_phased, g)
            if k in source_terms:
                rhs_p = rhs_p + source_terms[k]

            # Combined solve: F @ [X | x_p] = [-A_d | rhs_p]
            rhs_combined = np.concatenate(
                [-A_d, rhs_p[..., np.newaxis]], axis=-1
            )  # (nfreq, 4, 3)
            X_combined = np.linalg.solve(F, rhs_combined)  # (nfreq, 4, 3)
            Y = X_combined[:, :2, :2]
            g = X_combined[:, :2, 2]
        else:
            # Solve F @ X = -A_d for X, shape (nfreq, 4, 2)
            X = np.linalg.solve(F, -A_d)
            Y = X[:, :2, :]

    # --- Ocean-bottom extraction ---
    # Y is now Y_1, the Riccati matrix at the top of layer 1
    E_u_phased_1 = E_u[1][np.newaxis, :, :] * phase_d[1][:, np.newaxis, :]
    E1_eff = E_d[1][np.newaxis, :, :] + E_u_phased_1 @ Y  # (nfreq, 4, 2)

    q1 = None
    if has_sources:
        q1 = np.einsum("fij,fj->fi", E_u_phased_1, g)  # (nfreq, 4)
        if 0 in source_terms:
            q1 = q1 + source_terms[0]

    return _ocean_extraction_numpy(E1_eff, e_d_oc, e_u_oc, e0, nfreq, q1=q1)


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
    q1: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract reflectivity from E1_eff at the ocean bottom.

    Continuity conditions at ocean bottom (with particular solution q1):
      u_z:       e_u_oc[0]*U0P + e_d_oc[0]*e0 = E1_eff[row1,:] @ D1 + q1[1]
      sigma_zz:  e_u_oc[1]*U0P + e_d_oc[1]*e0 = E1_eff[row2,:] @ D1 + q1[2]
      sigma_xz:  0 = E1_eff[row3,:] @ D1 + q1[3]

    E-matrix rows: 0=u_x, 1=u_z, 2=sigma_zz, 3=sigma_xz

    When q1 is None (no sources), reduces to the standard surface-source case.

    Args:
        E1_eff: Effective eigenvector matrix at top of layer 1, shape (nfreq, 4, 2).
        e_d_oc: Ocean downgoing eigenvector, shape (2,).
        e_u_oc: Ocean upgoing eigenvector, shape (2,).
        e0: Ocean phase, shape (nfreq,).
        nfreq: Number of frequencies.
        q1: Particular-solution displacement-stress at ocean bottom, shape (nfreq, 4).
    """
    # Build 2x2 system, shape (nfreq, 2, 2)
    row_A = e_u_oc[0] * E1_eff[:, 2, :] - e_u_oc[1] * E1_eff[:, 1, :]  # (nfreq, 2)
    row_B = E1_eff[:, 3, :]  # (nfreq, 2)
    mat = np.stack([row_A, row_B], axis=1)  # (nfreq, 2, 2)

    # RHS, shape (nfreq, 2)
    rhs_A = (e_u_oc[0] * e_d_oc[1] - e_u_oc[1] * e_d_oc[0]) * e0  # (nfreq,)
    rhs_B = np.zeros(nfreq, dtype=_CDTYPE_NP)

    if q1 is not None:
        rhs_A = rhs_A - (e_u_oc[0] * q1[:, 2] - e_u_oc[1] * q1[:, 1])
        rhs_B = rhs_B - q1[:, 3]

    rhs = np.stack([rhs_A, rhs_B], axis=1)  # (nfreq, 2)

    # Solve for D1, shape (nfreq, 2)
    D1 = np.linalg.solve(mat, rhs[..., np.newaxis]).squeeze(-1)

    # Recover U0P from u_z continuity
    uz_match = np.einsum("fi,fi->f", E1_eff[:, 1, :], D1)
    if q1 is not None:
        U0_P = (uz_match + q1[:, 1] - e_d_oc[0] * e0) / e_u_oc[0]
    else:
        U0_P = (uz_match - e_d_oc[0] * e0) / e_u_oc[0]

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
    source_terms: dict[int, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-Riccati sweep for reflectivity (PyTorch, differentiable).

    Args:
        nlayer: Total number of layers.
        E_d: Downgoing eigenvector matrices, E_d[j] shape (4, 2) for j=1..nlayer-1.
        E_u: Upgoing eigenvector matrices, E_u[j] shape (4, 2) for j=1..nlayer-2.
        phase_d: Phase diagonals, phase_d[j] shape (nfreq, 2) for j=1..nlayer-2.
        e_d_oc: Ocean downgoing eigenvector, shape (2,).
        e_u_oc: Ocean upgoing eigenvector, shape (2,).
        e0: Ocean phase, shape (nfreq,).
        source_terms: Optional dict mapping interface index k to source tensor
            σ_k of shape (nfreq, 4). Produced by ``compute_source_vector``.

    Returns:
        (R, U0_P): Reflectivity and upgoing ocean amplitude, each shape (nfreq,).
    """
    nfreq = e0.shape[0]
    M = nlayer - 2

    if M == 0:
        return _two_layer_torch(E_d, e_d_oc, e_u_oc, e0, nfreq)

    Y = torch.zeros((nfreq, 2, 2), dtype=_CDTYPE_TORCH)

    if source_terms is None:
        source_terms = {}
    has_sources = len(source_terms) > 0
    if has_sources:
        g = torch.zeros((nfreq, 2), dtype=_CDTYPE_TORCH)

    for k in range(M, 0, -1):
        k_below = k + 1

        if k_below < nlayer - 1:
            E_u_phased = E_u[k_below].unsqueeze(0) * phase_d[k_below].unsqueeze(1)
            E_eff = E_d[k_below].unsqueeze(0) + E_u_phased @ Y
        else:
            E_u_phased = None
            E_eff = E_d[k_below].unsqueeze(0).expand(nfreq, 4, 2)

        A_d = E_d[k].unsqueeze(0) * phase_d[k].unsqueeze(1)

        E_u_k = E_u[k].unsqueeze(0).expand(nfreq, 4, 2)
        F = torch.cat([E_u_k, -E_eff], dim=-1)

        if has_sources:
            rhs_p = torch.zeros((nfreq, 4), dtype=_CDTYPE_TORCH)
            if E_u_phased is not None:
                rhs_p = rhs_p + torch.einsum("fij,fj->fi", E_u_phased, g)
            if k in source_terms:
                rhs_p = rhs_p + source_terms[k]

            rhs_combined = torch.cat(
                [-A_d, rhs_p.unsqueeze(-1)], dim=-1
            )  # (nfreq, 4, 3)
            X_combined = torch.linalg.solve(F, rhs_combined)
            Y = X_combined[:, :2, :2]
            g = X_combined[:, :2, 2]
        else:
            X = torch.linalg.solve(F, -A_d)
            Y = X[:, :2, :]

    # Ocean extraction
    E_u_phased_1 = E_u[1].unsqueeze(0) * phase_d[1].unsqueeze(1)
    E1_eff = E_d[1].unsqueeze(0) + E_u_phased_1 @ Y

    q1 = None
    if has_sources:
        q1 = torch.einsum("fij,fj->fi", E_u_phased_1, g)
        if 0 in source_terms:
            q1 = q1 + source_terms[0]

    return _ocean_extraction_torch(E1_eff, e_d_oc, e_u_oc, e0, nfreq, q1=q1)


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
    q1: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract reflectivity from E1_eff at ocean bottom (PyTorch)."""
    row_A = e_u_oc[0] * E1_eff[:, 2, :] - e_u_oc[1] * E1_eff[:, 1, :]
    row_B = E1_eff[:, 3, :]
    mat = torch.stack([row_A, row_B], dim=1)

    rhs_A = (e_u_oc[0] * e_d_oc[1] - e_u_oc[1] * e_d_oc[0]) * e0
    rhs_B = torch.zeros(nfreq, dtype=_CDTYPE_TORCH)

    if q1 is not None:
        rhs_A = rhs_A - (e_u_oc[0] * q1[:, 2] - e_u_oc[1] * q1[:, 1])
        rhs_B = rhs_B - q1[:, 3]

    rhs = torch.stack([rhs_A, rhs_B], dim=1)
    D1 = torch.linalg.solve(mat, rhs.unsqueeze(-1)).squeeze(-1)

    uz_match = torch.sum(E1_eff[:, 1, :] * D1, dim=-1)
    if q1 is not None:
        U0_P = (uz_match + q1[:, 1] - e_d_oc[0] * e0) / e_u_oc[0]
    else:
        U0_P = (uz_match - e_d_oc[0] * e0) / e_u_oc[0]

    R = e0 * U0_P
    return R, U0_P


# ---- Source placement helper ----


def compute_source_vector(
    S: np.ndarray,
    source_depth_frac: float,
    E_d_s: np.ndarray,
    E_u_s: np.ndarray,
    eta_s: complex,
    neta_s: complex,
    thickness_s: float,
    omega: np.ndarray,
    source_layer: int,
) -> dict[int, np.ndarray]:
    """Compute source vectors at the interfaces bounding the source layer.

    Given a jump discontinuity *S* at fractional depth within layer
    *source_layer*, decomposes *S* into upgoing/downgoing amplitudes and
    propagates them to the layer boundaries, producing the σ_k vectors
    that enter the Riccati sweep RHS (Chin et al. 1984 eq 4.4-4.5).

    Args:
        S: Jump vector at source depth, shape ``(nfreq, 4)``.
            Rows: ``[u_x, u_z, σ_zz/(−iω), σ_xz/(−iω)]``.
        source_depth_frac: Fractional depth within the layer (0 = top, 1 = bottom).
        E_d_s: Downgoing eigenvectors of the source layer, shape ``(4, 2)``.
        E_u_s: Upgoing eigenvectors of the source layer, shape ``(4, 2)``.
        eta_s: P-wave vertical slowness in the source layer.
        neta_s: S-wave vertical slowness in the source layer.
        thickness_s: Source layer thickness.
        omega: Angular frequencies, shape ``(nfreq,)``.
        source_layer: 1-based index of the source layer (must be a finite
            elastic layer, i.e. ``1 ≤ source_layer ≤ nlayer − 2``).

    Returns:
        Dict mapping interface index *k* to source vector σ_k of shape
        ``(nfreq, 4)``.  Key ``source_layer − 1`` is the top interface
        (or ocean bottom when ``source_layer == 1``); key ``source_layer``
        is the bottom interface.  Signs are pre-applied so the caller can
        pass the dict directly to :func:`riccati_sweep_numpy`.
    """
    f = source_depth_frac
    h = thickness_s

    # Full 4×4 eigenvector matrix [E_d | E_u] and its inverse
    E_full = np.concatenate([E_d_s, E_u_s], axis=1)  # (4, 4)
    E_inv = np.linalg.inv(E_full)  # (4, 4)

    # Decompose jump: [c_d; c_u] = E_inv @ S^T  →  coeffs (nfreq, 4)
    coeffs = S @ E_inv.T
    c_d = coeffs[:, :2]  # downgoing amplitudes at source depth
    c_u = coeffs[:, 2:]  # upgoing amplitudes at source depth

    # Phase propagation within the layer
    phase_top = np.stack(
        [
            np.exp(1j * omega * eta_s * f * h),
            np.exp(1j * omega * neta_s * f * h),
        ],
        axis=-1,
    )  # (nfreq, 2) — upgoing from source to top

    phase_bot = np.stack(
        [
            np.exp(1j * omega * eta_s * (1.0 - f) * h),
            np.exp(1j * omega * neta_s * (1.0 - f) * h),
        ],
        axis=-1,
    )  # (nfreq, 2) — downgoing from source to bottom

    source_terms: dict[int, np.ndarray] = {}

    # σ_top: upgoing from source propagated to top of layer (+ve, right-side)
    c_u_top = c_u * phase_top  # (nfreq, 2)
    sigma_top = c_u_top @ E_u_s.T  # (nfreq, 4)
    source_terms[source_layer - 1] = sigma_top

    # σ_bot: downgoing from source propagated to bottom of layer (−ve, left-side)
    c_d_bot = c_d * phase_bot  # (nfreq, 2)
    sigma_bot = -(c_d_bot @ E_d_s.T)  # (nfreq, 4)
    source_terms[source_layer] = sigma_bot

    return source_terms
