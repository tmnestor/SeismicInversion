"""Layered-medium Green's function via two-pass Riccati algorithm.

Computes the full 4x4 (P-SV) and 2x2 (SH) displacement-traction Green's
function in the horizontal-wavenumber domain. Unlike the existing reflectivity
solver (which returns a scalar R at the ocean surface), this module returns
the complete response at an arbitrary receiver interface.

Vectorises over **horizontal wavenumber kH** at fixed **frequency omega**.

Convention: exp(-iwt) inverse Fourier transform, depth positive downward.
"""

import numpy as np

from Kennett_Reflectivity.layer_model import LayerModel

from .layer_matrix import (
    layer_eigenvectors_batched,
    layer_eigenvectors_sh_batched,
    ocean_eigenvectors_batched,
)

__all__ = [
    "assemble_greens_6x6",
    "layered_greens_6x6",
    "layered_greens_psv",
    "layered_greens_sh",
    "riccati_greens_psv",
    "riccati_greens_sh",
]

_CD = np.complex128


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _vertical_slowness_batched(slowness: complex, p: np.ndarray) -> np.ndarray:
    """Vertical slowness with Im(eta) > 0 branch cut, vectorised.

    Args:
        slowness: Complex slowness of the layer (scalar).
        p: Horizontal slowness array, shape (n_kH,).

    Returns:
        Vertical slowness eta, shape (n_kH,).
    """
    T = (slowness + p) * (slowness - p)
    eta = np.sqrt(T)
    # Enforce Im(eta) > 0
    mask = eta.imag <= 0.0
    eta[mask] = -eta[mask]
    return eta


def _prepare_model_arrays(
    model: LayerModel,
    omega: complex,
    kH: np.ndarray,
) -> tuple[
    int,
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Precompute batched eigenvectors and phases for all layers.

    Returns:
        nlayer, E_d, E_u, phase_d, e_d_oc, e_u_oc, e0
    """
    nlayer = model.n_layers
    s_p = model.complex_slowness_p()
    s_s = model.complex_slowness_s()
    beta_c = model.complex_velocity_s()

    # p = kH / omega for each wavenumber — shape (n_kH,)
    p = kH / omega

    # Vertical slownesses per layer — shape (n_kH,)
    eta = {}
    neta = {}
    for j in range(nlayer):
        eta[j] = _vertical_slowness_batched(s_p[j], p)
    for j in range(1, nlayer):
        neta[j] = _vertical_slowness_batched(s_s[j], p)

    # Eigenvectors — batched over kH
    E_d: dict[int, np.ndarray] = {}
    E_u: dict[int, np.ndarray] = {}
    for j in range(1, nlayer):
        E_d[j], E_u[j] = layer_eigenvectors_batched(
            p, eta[j], neta[j], model.rho[j], beta_c[j]
        )

    # Ocean eigenvectors
    e_d_oc, e_u_oc = ocean_eigenvectors_batched(p, eta[0], model.rho[0])

    # Ocean phase: exp(i * omega * eta_0 * h_0), shape (n_kH,)
    e0 = np.exp(1j * omega * eta[0] * model.thickness[0])

    # Phase diagonals for finite elastic layers
    phase_d: dict[int, np.ndarray] = {}
    for j in range(1, nlayer - 1):
        ph_p = np.exp(1j * omega * eta[j] * model.thickness[j])
        ph_s = np.exp(1j * omega * neta[j] * model.thickness[j])
        phase_d[j] = np.stack([ph_p, ph_s], axis=-1)  # (n_kH, 2)

    return nlayer, E_d, E_u, phase_d, e_d_oc, e_u_oc, e0


# ---------------------------------------------------------------------------
# P-SV two-pass Riccati Green's function
# ---------------------------------------------------------------------------


def riccati_greens_psv(
    nlayer: int,
    E_d: dict[int, np.ndarray],
    E_u: dict[int, np.ndarray],
    phase_d: dict[int, np.ndarray],
    e_d_oc: np.ndarray,
    e_u_oc: np.ndarray,
    e0: np.ndarray,
    source_iface: int,
    receiver_iface: int,
) -> np.ndarray:
    """P-SV Green's function via two-pass Riccati algorithm.

    Args:
        nlayer: Total number of layers (ocean + elastic + half-space).
        E_d: Downgoing eigenvectors, E_d[j] shape (n_kH, 4, 2) for j=1..nlayer-1.
        E_u: Upgoing eigenvectors, E_u[j] shape (n_kH, 4, 2) for j=1..nlayer-2.
        phase_d: Phase diagonals, phase_d[j] shape (n_kH, 2) for j=1..nlayer-2.
        e_d_oc: Ocean downgoing eigenvector, shape (n_kH, 2).
        e_u_oc: Ocean upgoing eigenvector, shape (n_kH, 2).
        e0: Ocean phase, shape (n_kH,).
        source_iface: Interface index where unit source is placed (0..M).
            0 = ocean bottom, M = bottom of last finite elastic layer.
        receiver_iface: Interface index where Green's function is evaluated.

    Returns:
        G_psv: shape (n_kH, 4, 4) — Green's function in Riccati basis
            [u_x, u_z, sigma_zz/(-iw), sigma_xz/(-iw)].
    """
    n_kH = e0.shape[0]
    M = nlayer - 2  # number of finite elastic layers

    # ------------------------------------------------------------------
    # Pass 1: Upward sweep — store Y_k, g_k at all interfaces
    # ------------------------------------------------------------------
    Y = {}  # Y[k]: (n_kH, 2, 2)
    g = {}  # g[k]: (n_kH, 2, 4)

    # Radiation condition at bottom
    Y[M + 1] = np.zeros((n_kH, 2, 2), dtype=_CD)
    g[M + 1] = np.zeros((n_kH, 2, 4), dtype=_CD)

    for k in range(M, 0, -1):
        k_below = k + 1

        # Build E_eff = E_d[k_below] + E_u[k_below] * diag(phase) @ Y
        if k_below < nlayer - 1:
            # k_below is a finite elastic layer
            E_u_phased = E_u[k_below] * phase_d[k_below][:, np.newaxis, :]
            E_eff = E_d[k_below] + E_u_phased @ Y[k_below]
        else:
            # k_below is the half-space — no upgoing
            E_u_phased = None
            E_eff = E_d[k_below].copy()

        # A_d = E_d[k] * diag(phase[k])
        A_d = E_d[k] * phase_d[k][:, np.newaxis, :]  # (n_kH, 4, 2)

        # F = [E_u[k], -E_eff]
        F = np.concatenate([E_u[k], -E_eff], axis=-1)  # (n_kH, 4, 4)

        # Particular RHS: propagated g from below + source injection
        rhs_p = np.zeros((n_kH, 4, 4), dtype=_CD)
        if E_u_phased is not None:
            rhs_p += np.einsum("fij,fjk->fik", E_u_phased, g[k_below])
        if k == source_iface:
            rhs_p += np.eye(4, dtype=_CD)[np.newaxis, :, :]

        # Combined solve: F @ X = [-A_d | rhs_p]
        rhs = np.concatenate([-A_d, rhs_p], axis=-1)  # (n_kH, 4, 6)
        X = np.linalg.solve(F, rhs)  # (n_kH, 4, 6)
        Y[k] = X[:, :2, :2]  # (n_kH, 2, 2)
        g[k] = X[:, :2, 2:]  # (n_kH, 2, 4)

    # ------------------------------------------------------------------
    # Ocean extraction: determine D1 (shape n_kH, 2, 4)
    # ------------------------------------------------------------------
    E_u_phased_1 = E_u[1] * phase_d[1][:, np.newaxis, :]
    E1_eff = E_d[1] + E_u_phased_1 @ Y[1]  # (n_kH, 4, 2)

    # Particular solution at ocean bottom
    q1 = np.einsum("fij,fjk->fik", E_u_phased_1, g[1])  # (n_kH, 4, 4)
    if 0 == source_iface:
        q1 = q1 + np.eye(4, dtype=_CD)[np.newaxis, :, :]

    # 2x2 system per source column (matrix RHS):
    # row_A: e_u_oc[0]*sigma_zz_row - e_u_oc[1]*u_z_row
    # row_B: sigma_xz_row
    # Each operates on the 2 D1-components, with 4 source columns on RHS
    row_A = (
        e_u_oc[:, 0:1] * E1_eff[:, 2, :] - e_u_oc[:, 1:2] * E1_eff[:, 1, :]
    )  # (n_kH, 2)
    row_B = E1_eff[:, 3, :]  # (n_kH, 2)
    mat = np.stack([row_A, row_B], axis=1)  # (n_kH, 2, 2)

    # RHS for the 4 source columns
    rhs_A = (
        e_u_oc[:, 0] * e_d_oc[:, 1] - e_u_oc[:, 1] * e_d_oc[:, 0]
    ) * e0  # (n_kH,) — ocean source term (only if source_iface includes ocean)
    # For Green's function, the ocean source is NOT active (source is at
    # source_iface in the solid). The ocean excitation comes from
    # the particular solution q1.
    # rhs = -q1 contribution
    rhs_A_mat = -(e_u_oc[:, 0:1] * q1[:, 2, :] - e_u_oc[:, 1:2] * q1[:, 1, :])
    rhs_B_mat = -q1[:, 3, :]  # (n_kH, 4)
    rhs_mat = np.stack([rhs_A_mat, rhs_B_mat], axis=1)  # (n_kH, 2, 4)

    D1 = np.linalg.solve(mat, rhs_mat)  # (n_kH, 2, 4)

    # ------------------------------------------------------------------
    # Pass 2: Downward sweep — propagate amplitudes to receiver
    # ------------------------------------------------------------------
    # D[k]: downgoing amplitudes at top of layer k, shape (n_kH, 2, 4)
    # U[k]: upgoing amplitudes at bottom of layer k, shape (n_kH, 2, 4)

    def _state_at_top(k: int, D_k: np.ndarray) -> np.ndarray:
        """State at top of layer k: E_d @ D + E_u*phase @ U."""
        U_k = Y[k] @ D_k + g[k]
        E_u_ph = E_u[k] * phase_d[k][:, np.newaxis, :]
        return np.einsum("fij,fjk->fik", E_d[k], D_k) + np.einsum(
            "fij,fjk->fik", E_u_ph, U_k
        )

    def _state_at_bottom(k: int, D_k: np.ndarray) -> np.ndarray:
        """State at bottom of layer k: E_d*phase @ D + E_u @ U."""
        U_k = Y[k] @ D_k + g[k]
        D_ph = D_k * phase_d[k][:, :, np.newaxis]
        return np.einsum("fij,fjk->fik", E_d[k], D_ph) + np.einsum(
            "fij,fjk->fik", E_u[k], U_k
        )

    # Receiver at ocean bottom: state at top of layer 1
    if receiver_iface == 0:
        return _state_at_top(1, D1)

    D = D1
    for k in range(1, receiver_iface + 1):
        if k == receiver_iface:
            # State at bottom of layer k (from the layer-k side, above the jump)
            return _state_at_bottom(k, D)

        # State at bottom of layer k, then subtract source to get
        # state at top of layer k+1 (convention: bot_k - sigma = top_k+1)
        state = _state_at_bottom(k, D)
        if k == source_iface:
            state = state - np.eye(4, dtype=_CD)[np.newaxis, :, :]

        k1 = k + 1
        if k1 < nlayer - 1:
            # Finite layer: [E_d, E_u*phase] @ [D; U] = state_top_k+1
            E_u_ph = E_u[k1] * phase_d[k1][:, np.newaxis, :]
            F_k1 = np.concatenate([E_d[k1], E_u_ph], axis=-1)  # (n_kH, 4, 4)
            x = np.linalg.solve(F_k1, state)  # (n_kH, 4, 4)
            D = x[:, :2, :]  # D at top of layer k+1
        else:
            # Half-space: return state directly (no upgoing)
            return state

    # Should not reach here
    return _state_at_bottom(receiver_iface, D)


def layered_greens_psv(
    model: LayerModel,
    omega: complex,
    kH: np.ndarray,
    source_iface: int,
    receiver_iface: int,
) -> np.ndarray:
    """P-SV Green's function in the Riccati basis.

    Convenience wrapper that builds eigenvectors from a LayerModel.

    Args:
        model: Stratified elastic model.
        omega: Angular frequency (scalar, may be complex for damping).
        kH: Horizontal wavenumber magnitudes, shape (n_kH,).
        source_iface: Interface index for source (0 = ocean bottom).
        receiver_iface: Interface index for receiver.

    Returns:
        G_psv: shape (n_kH, 4, 4) in basis [u_x, u_z, sigma_zz/(-iw), sigma_xz/(-iw)].
    """
    nlayer, E_d, E_u, phase_d, e_d_oc, e_u_oc, e0 = _prepare_model_arrays(
        model, omega, kH
    )
    return riccati_greens_psv(
        nlayer,
        E_d,
        E_u,
        phase_d,
        e_d_oc,
        e_u_oc,
        e0,
        source_iface,
        receiver_iface,
    )


# ---------------------------------------------------------------------------
# SH two-pass Riccati Green's function
# ---------------------------------------------------------------------------


def _prepare_sh_arrays(
    model: LayerModel,
    omega: complex,
    kH: np.ndarray,
) -> tuple[
    int,
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
]:
    """Precompute SH eigenvectors and phases.

    Returns:
        nlayer, e_d_sh, e_u_sh, phase_sh
    """
    nlayer = model.n_layers
    s_s = model.complex_slowness_s()
    beta_c = model.complex_velocity_s()
    p = kH / omega

    e_d_sh: dict[int, np.ndarray] = {}
    e_u_sh: dict[int, np.ndarray] = {}
    phase_sh: dict[int, np.ndarray] = {}

    for j in range(1, nlayer):
        neta_j = _vertical_slowness_batched(s_s[j], p)
        e_d_sh[j], e_u_sh[j] = layer_eigenvectors_sh_batched(
            neta_j, model.rho[j], beta_c[j]
        )
        if j < nlayer - 1:
            phase_sh[j] = np.exp(1j * omega * neta_j * model.thickness[j])  # (n_kH,)

    return nlayer, e_d_sh, e_u_sh, phase_sh


def riccati_greens_sh(
    nlayer: int,
    e_d_sh: dict[int, np.ndarray],
    e_u_sh: dict[int, np.ndarray],
    phase_sh: dict[int, np.ndarray],
    source_iface: int,
    receiver_iface: int,
) -> np.ndarray:
    """SH Green's function via two-pass scalar Riccati algorithm.

    The SH system has 2 components (u_t, sigma_tz/(-iw)) and 1 wave type
    per direction, so the Riccati variable is scalar (1x1).

    Ocean layer is ignored for SH (no shear in fluid). The free-surface
    condition sigma_tz = 0 at the ocean bottom is enforced.

    Args:
        nlayer: Total layers (ocean + elastic + half-space).
        e_d_sh: Downgoing SH eigenvectors, e_d_sh[j] shape (n_kH, 2, 1).
        e_u_sh: Upgoing SH eigenvectors, e_u_sh[j] shape (n_kH, 2, 1).
        phase_sh: SH phase factors, phase_sh[j] shape (n_kH,).
        source_iface: Interface index (1..M for solid-solid interfaces).
        receiver_iface: Interface index for receiver.

    Returns:
        G_sh: shape (n_kH, 2, 2) in basis [u_t, sigma_tz/(-iw)].
    """
    n_kH = e_d_sh[1].shape[0]
    M = nlayer - 2

    # ------------------------------------------------------------------
    # Pass 1: Upward sweep (scalar Riccati)
    # ------------------------------------------------------------------
    # Y[k]: scalar (n_kH,), g[k]: (n_kH, 1, 2) — 2 source columns
    Y_sh: dict[int, np.ndarray] = {}
    g_sh: dict[int, np.ndarray] = {}

    Y_sh[M + 1] = np.zeros(n_kH, dtype=_CD)
    g_sh[M + 1] = np.zeros((n_kH, 1, 2), dtype=_CD)

    for k in range(M, 0, -1):
        k_below = k + 1

        if k_below < nlayer - 1:
            ph_below = phase_sh[k_below][:, np.newaxis, np.newaxis]  # (n_kH,1,1)
            e_u_phased = e_u_sh[k_below] * ph_below  # (n_kH, 2, 1)
            e_eff = (
                e_d_sh[k_below] + e_u_phased * Y_sh[k_below][:, np.newaxis, np.newaxis]
            )
        else:
            e_u_phased = None
            e_eff = e_d_sh[k_below].copy()

        # A_d = e_d_sh[k] * phase_sh[k]
        ph_k = phase_sh[k][:, np.newaxis, np.newaxis]
        a_d = e_d_sh[k] * ph_k  # (n_kH, 2, 1)

        # F = [e_u_sh[k], -e_eff], shape (n_kH, 2, 2)
        F = np.concatenate([e_u_sh[k], -e_eff], axis=-1)

        # Particular RHS
        rhs_p = np.zeros((n_kH, 2, 2), dtype=_CD)
        if e_u_phased is not None:
            rhs_p += np.einsum("fij,fjk->fik", e_u_phased, g_sh[k_below])
        if k == source_iface:
            rhs_p += np.eye(2, dtype=_CD)[np.newaxis, :, :]

        # Combined: F @ X = [-a_d | rhs_p], X shape (n_kH, 2, 3)
        rhs = np.concatenate([-a_d, rhs_p], axis=-1)
        X = np.linalg.solve(F, rhs)  # (n_kH, 2, 3)
        Y_sh[k] = X[:, 0, 0]  # scalar Riccati variable
        g_sh[k] = X[:, :1, 1:]  # (n_kH, 1, 2)

    # ------------------------------------------------------------------
    # Surface extraction: sigma_tz = 0 at ocean bottom
    # ------------------------------------------------------------------
    # At the ocean bottom / top of layer 1:
    # e_eff_1 = e_d_sh[1] + e_u_sh[1]*phase[1]*Y[1]
    ph_1 = phase_sh[1][:, np.newaxis, np.newaxis]
    e_u_ph_1 = e_u_sh[1] * ph_1
    e_eff_1 = e_d_sh[1] + e_u_ph_1 * Y_sh[1][:, np.newaxis, np.newaxis]

    q1_sh = np.einsum("fij,fjk->fik", e_u_ph_1, g_sh[1])  # (n_kH, 2, 2)
    if source_iface == 0:
        q1_sh = q1_sh + np.eye(2, dtype=_CD)[np.newaxis, :, :]

    # sigma_tz = 0 => e_eff_1[1,0] * D1 + q1_sh[1,:] = 0
    # D1 = -q1_sh[1,:] / e_eff_1[1,0], shape (n_kH, 1, 2)
    D1_sh = -q1_sh[:, 1:2, :] / e_eff_1[:, 1:2, :]  # (n_kH, 1, 2)

    # ------------------------------------------------------------------
    # Pass 2: Downward sweep to receiver
    # ------------------------------------------------------------------
    if receiver_iface == 0:
        G_sh = np.einsum("fij,fjk->fik", e_eff_1, D1_sh) + q1_sh
        return G_sh

    def _sh_state_at_top(k: int, D_k: np.ndarray) -> np.ndarray:
        """SH state at top of layer k."""
        U_k = Y_sh[k][:, np.newaxis, np.newaxis] * D_k + g_sh[k]
        ph = phase_sh[k][:, np.newaxis, np.newaxis]
        return np.einsum("fij,fjk->fik", e_d_sh[k], D_k) + np.einsum(
            "fij,fjk->fik", e_u_sh[k] * ph, U_k
        )

    def _sh_state_at_bottom(k: int, D_k: np.ndarray) -> np.ndarray:
        """SH state at bottom of layer k."""
        U_k = Y_sh[k][:, np.newaxis, np.newaxis] * D_k + g_sh[k]
        ph = phase_sh[k][:, np.newaxis, np.newaxis]
        return np.einsum("fij,fjk->fik", e_d_sh[k] * ph, D_k) + np.einsum(
            "fij,fjk->fik", e_u_sh[k], U_k
        )

    D = D1_sh
    for k in range(1, receiver_iface + 1):
        if k == receiver_iface:
            return _sh_state_at_bottom(k, D)

        # State at bottom of layer k, subtract source to get top of k+1
        state = _sh_state_at_bottom(k, D)
        if k == source_iface:
            state = state - np.eye(2, dtype=_CD)[np.newaxis, :, :]

        k1 = k + 1
        if k1 < nlayer - 1:
            # [e_d, e_u*phase] @ [D; U] = state  — 2×2 solve
            ph_k1 = phase_sh[k1][:, np.newaxis, np.newaxis]
            F_k1 = np.concatenate(
                [e_d_sh[k1], e_u_sh[k1] * ph_k1], axis=-1
            )  # (n_kH, 2, 2)
            x = np.linalg.solve(F_k1, state)  # (n_kH, 2, 2)
            D = x[:, :1, :]  # D at top of layer k+1, shape (n_kH, 1, 2)
        else:
            # Half-space: return state directly
            return state

    # Should not reach here
    return _sh_state_at_bottom(receiver_iface, D)


def layered_greens_sh(
    model: LayerModel,
    omega: complex,
    kH: np.ndarray,
    source_iface: int,
    receiver_iface: int,
) -> np.ndarray:
    """SH Green's function convenience wrapper.

    Args:
        model: Stratified elastic model.
        omega: Angular frequency (scalar).
        kH: Horizontal wavenumber magnitudes, shape (n_kH,).
        source_iface: Interface index for source.
        receiver_iface: Interface index for receiver.

    Returns:
        G_sh: shape (n_kH, 2, 2) in basis [u_t, sigma_tz/(-iw)].
    """
    nlayer, e_d_sh, e_u_sh, phase_sh = _prepare_sh_arrays(model, omega, kH)
    return riccati_greens_sh(
        nlayer,
        e_d_sh,
        e_u_sh,
        phase_sh,
        source_iface,
        receiver_iface,
    )


# ---------------------------------------------------------------------------
# 6x6 Assembly: P-SV + SH → Cartesian
# ---------------------------------------------------------------------------


def assemble_greens_6x6(
    G_psv: np.ndarray,
    G_sh: np.ndarray,
    cos_phi: np.ndarray,
    sin_phi: np.ndarray,
    omega: complex,
) -> np.ndarray:
    """Assemble full 6x6 Green's function from P-SV and SH components.

    Combines the 4x4 P-SV Green's function (in sagittal plane) with the
    2x2 SH Green's function, applies azimuthal rotation, and converts
    from Riccati convention to physical Cartesian convention.

    Output convention: (u_z, u_x, u_y, sigma_zz, sigma_xz, sigma_yz).

    Args:
        G_psv: P-SV Green's function, shape (..., 4, 4).
            Riccati basis: [u_x, u_z, sigma_zz/(-iw), sigma_xz/(-iw)].
        G_sh: SH Green's function, shape (..., 2, 2).
            Basis: [u_t, sigma_tz/(-iw)].
        cos_phi: cos(azimuth), shape (...,).
        sin_phi: sin(azimuth), shape (...,).
        omega: Angular frequency (scalar).

    Returns:
        G6: shape (..., 6, 6) in Cartesian basis
            (u_z, u_x, u_y, sigma_zz, sigma_xz, sigma_yz).
    """
    shape = G_psv.shape[:-2]
    G6 = np.zeros(shape + (6, 6), dtype=_CD)

    # Permute P-SV: Riccati [u_x, u_z, σ_zz/(−iω), σ_xz/(−iω)]
    #            → physical [u_z, u_r, σ_zz, σ_rz]
    # Indices: Riccati 0→u_x=u_r, 1→u_z, 2→σ_zz/(−iω), 3→σ_xz/(−iω)=σ_rz/(−iω)
    # Physical: 0→u_z (from Riccati 1), 1→u_r (from Riccati 0),
    #           2→σ_zz (Riccati 2 * (-iω)), 3→σ_rz (Riccati 3 * (-iω))
    perm = [1, 0, 2, 3]
    G_psv_perm = G_psv[..., perm, :][..., :, perm]

    # Scale stress rows and columns by (-iω)
    miw = -1j * omega
    # Stress is in rows 2,3 and columns 2,3
    G_psv_perm[..., 2:, :] *= miw  # stress rows (output)
    G_psv_perm[..., :2, 2:] *= miw  # stress columns (source)

    # P-SV fills the (u_z, u_r, σ_zz, σ_rz) block
    # In our 6x6: indices 0=u_z, 1=u_x, 2=u_y, 3=σ_zz, 4=σ_xz, 5=σ_yz
    # P-SV sagittal → indices (0, r, 3, r_stress) where r is radial
    # u_z, σ_zz are index 0, 3 — azimuth-independent
    # u_r, σ_rz decompose into u_x*cos + u_y*sin

    # SH: [u_t, σ_tz/(−iω)] → physical [u_t, σ_tz]
    G_sh_phys = G_sh.copy()
    G_sh_phys[..., 1:, :] *= miw  # stress row
    G_sh_phys[..., :1, 1:] *= miw  # stress column

    c = cos_phi[..., np.newaxis, np.newaxis]
    s = sin_phi[..., np.newaxis, np.newaxis]

    # Build rotation from (z, r, t) to (z, x, y):
    # u_z stays, u_x = u_r*cos - u_t*sin, u_y = u_r*sin + u_t*cos
    # Same for stresses

    # P-SV physical: rows/cols (u_z=0, u_r=1, σ_zz=2, σ_rz=3)
    # SH physical: rows/cols (u_t=0, σ_tz=1)
    # 6x6 output: (u_z=0, u_x=1, u_y=2, σ_zz=3, σ_xz=4, σ_yz=5)

    # Map P-SV (z, r, σ_zz, σ_rz) → (0, 1, 3, 4) in 6x6
    psv_idx = [0, 1, 3, 4]
    # Map SH (t, σ_tz) → (2, 5) in 6x6
    sh_idx = [2, 5]

    # Intermediate: place sagittal (z,r) and transverse (t) into (z, r, t)
    # Then rotate (r, t) → (x, y) via rotation matrix
    # For the displacement-displacement block (and stress-stress):
    #   [u_x]   [cos  -sin] [u_r]
    #   [u_y] = [sin   cos] [u_t]
    # Applied to both rows and columns of G

    # P-SV contributes to (z,r) × (z,r), SH contributes to (t) × (t)
    # In the rotated frame:
    # G_xx = cos*G_rr*cos + sin*G_tt*sin  (cross terms vanish by isotropy)
    # But actually for a general source/receiver we need the full rotation.

    # Simpler: build 3x3 sagittal-block in (z, r, t) then rotate
    # For displacement (3x3): indices z=0, r=1, t=2
    # G_zrt[0,0] = G_psv[0,0], G_zrt[0,1] = G_psv[0,1], G_zrt[1,0] = G_psv[1,0]
    # G_zrt[1,1] = G_psv[1,1], G_zrt[2,2] = G_sh[0,0]
    # Cross terms G_zrt[0,2], G_zrt[2,0], G_zrt[1,2], G_zrt[2,1] = 0

    # Build 6x6 in (z, r, t, σ_zz, σ_rz, σ_tz) then rotate
    G_zrt = np.zeros(shape + (6, 6), dtype=_CD)
    for i, pi in enumerate(psv_idx):
        for j, pj in enumerate(psv_idx):
            G_zrt[..., pi, pj] = G_psv_perm[..., i, j]
    for i, si in enumerate(sh_idx):
        for j, sj in enumerate(sh_idx):
            G_zrt[..., si, sj] = G_sh_phys[..., i, j]

    # Rotation: R @ G_zrt @ R^T where R acts on (x, y) from (r, t)
    # R is block-diagonal: z and σ_zz untouched, (r,t)→(x,y), (σ_rz,σ_tz)→(σ_xz,σ_yz)
    # Build 6x6 rotation matrix
    R = np.zeros(shape + (6, 6), dtype=_CD)
    R[..., 0, 0] = 1.0  # z → z
    R[..., 3, 3] = 1.0  # σ_zz → σ_zz
    # (r, t) → (x, y): indices 1,2 and 4,5
    R[..., 1, 1] = cos_phi  # x from r
    R[..., 1, 2] = -sin_phi  # x from t
    R[..., 2, 1] = sin_phi  # y from r
    R[..., 2, 2] = cos_phi  # y from t
    R[..., 4, 4] = cos_phi
    R[..., 4, 5] = -sin_phi
    R[..., 5, 4] = sin_phi
    R[..., 5, 5] = cos_phi

    G6 = np.einsum("...ij,...jk,...lk->...il", R, G_zrt, R)

    return G6


def layered_greens_6x6(
    model: LayerModel,
    omega: complex,
    kx: np.ndarray,
    ky: np.ndarray,
    source_iface: int,
    receiver_iface: int,
) -> np.ndarray:
    """Full 6x6 Green's function on a 2D (kx, ky) grid.

    Combines P-SV and SH Riccati Green's functions with azimuthal rotation
    to produce the Cartesian displacement-traction Green's function.

    Args:
        model: Stratified elastic model.
        omega: Angular frequency (scalar, may be complex).
        kx: Horizontal wavenumber x-component, arbitrary shape.
        ky: Horizontal wavenumber y-component, same shape as kx.
        source_iface: Interface index for source (0 = ocean bottom).
        receiver_iface: Interface index for receiver.

    Returns:
        G6: shape (*kx.shape, 6, 6) in Cartesian basis
            (u_z, u_x, u_y, sigma_zz, sigma_xz, sigma_yz).
    """
    orig_shape = kx.shape
    kx_flat = kx.ravel()
    ky_flat = ky.ravel()

    kH = np.sqrt(kx_flat**2 + ky_flat**2)

    # Azimuthal angle
    phi = np.arctan2(ky_flat, kx_flat)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    # Handle kH = 0 (set small floor to avoid division by zero in p = kH/omega)
    kH_safe = np.where(kH > 0, kH, 1e-30)

    G_psv = layered_greens_psv(model, omega, kH_safe, source_iface, receiver_iface)
    G_sh = layered_greens_sh(model, omega, kH_safe, source_iface, receiver_iface)

    G6 = assemble_greens_6x6(G_psv, G_sh, cos_phi, sin_phi, omega)

    return G6.reshape(orig_shape + (6, 6))
