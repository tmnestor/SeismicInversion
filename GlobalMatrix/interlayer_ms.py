"""Interlayer-only multiple scattering for layered elastic media.

Computes the reflectivity perturbation from scatterers embedded at layer
interfaces, retaining only interlayer (vertical) multiple scattering paths
via the wavenumber-domain Riccati Green's function.  Intralayer (side)
scattering is neglected — the research question is when this is valid.

Algorithm (per ω, kH):
    1. Background incident field from ocean P-wave source.
    2. Interlayer Green's matrix G_block[i,j] for i≠j scatterer interfaces.
    3. Foldy-Lax solve: (I − G_block @ T_block) ψ = ψ⁰.
    4. Scattered reflectivity via ocean extraction.

Convention: exp(-iωt), depth positive downward, state vector
    [u_x, u_z, σ_zz/(−iω), σ_xz/(−iω)].
"""

from dataclasses import dataclass

import numpy as np

from Kennett_Reflectivity.layer_model import LayerModel

from .layered_greens import (
    _interface_elastic_properties,
    _prepare_model_arrays,
    layered_greens_6x6,
    layered_greens_9x9,
    layered_greens_psv,
    riccati_greens_psv,
    strain_from_displacement_traction,
    traction_from_strain,
)

__all__ = [
    "InterlayerMSResult",
    "InterlayerMSResult9x9",
    "ScattererSlab",
    "ScattererSlab9x9",
    "background_incident_field",
    "background_incident_field_9x9",
    "build_interlayer_greens_matrix",
    "build_interlayer_greens_matrix_9x9",
    "interlayer_ms_reflectivity",
    "interlayer_ms_reflectivity_9x9",
    "scattered_reflectivity",
    "scattered_reflectivity_9x9",
    "solve_interlayer_foldy_lax",
    "tmatrix_6x6_to_4x4_psv",
    "tmatrix_9x9_to_4x4_psv",
]

_CD = np.complex128


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ScattererSlab:
    """Layered model with embedded scatterers at specific interfaces.

    Args:
        model: Background stratified elastic model.
        scatterer_ifaces: Interface indices containing scatterers.
            Must be in range [1, M] where M = n_layers - 2.
        tmatrices: Mapping from interface index to (4, 4) P-SV T-matrix.
        number_densities: Mapping from interface index to areal number
            density (scatterers per unit area).
    """

    model: LayerModel
    scatterer_ifaces: list[int]
    tmatrices: dict[int, np.ndarray]
    number_densities: dict[int, float]

    def __post_init__(self) -> None:
        """Validate scatterer configuration."""
        M = self.model.n_layers - 2  # number of finite elastic layers
        for idx in self.scatterer_ifaces:
            if not (1 <= idx <= M):
                msg = (
                    f"Scatterer interface {idx} out of range [1, {M}] "
                    f"for a model with {self.model.n_layers} layers"
                )
                raise ValueError(msg)
            if idx not in self.tmatrices:
                msg = f"Missing T-matrix for scatterer interface {idx}"
                raise ValueError(msg)
            T = self.tmatrices[idx]
            if T.shape != (4, 4):
                msg = (
                    f"T-matrix at interface {idx} has shape {T.shape}, expected (4, 4)"
                )
                raise ValueError(msg)
            if idx not in self.number_densities:
                msg = f"Missing number density for scatterer interface {idx}"
                raise ValueError(msg)


@dataclass
class InterlayerMSResult:
    """Result of interlayer multiple scattering calculation.

    Args:
        R_background: Background reflectivity (no scatterers), shape (n_kH,).
        R_total: Total reflectivity with interlayer MS, shape (n_kH,).
        R_born: Born (single-scattering) approximation, shape (n_kH,).
        psi_exciting: Exciting field at each scatterer interface,
            dict mapping iface → (n_kH, 4).
        psi_incident: Incident field at each scatterer interface,
            dict mapping iface → (n_kH, 4).
    """

    R_background: np.ndarray
    R_total: np.ndarray
    R_born: np.ndarray
    psi_exciting: dict[int, np.ndarray]
    psi_incident: dict[int, np.ndarray]


@dataclass
class ScattererSlab9x9:
    """Layered model with scatterers using 9×9 (u, ε) T-matrices.

    Args:
        model: Background stratified elastic model.
        scatterer_ifaces: Interface indices containing scatterers.
            Must be in range [1, M] where M = n_layers - 2.
        tmatrices: Mapping from interface index to (9, 9) T-matrix
            in the displacement-strain basis.
        number_densities: Mapping from interface index to areal number
            density (scatterers per unit area).
    """

    model: LayerModel
    scatterer_ifaces: list[int]
    tmatrices: dict[int, np.ndarray]
    number_densities: dict[int, float]

    def __post_init__(self) -> None:
        """Validate scatterer configuration."""
        M = self.model.n_layers - 2
        for idx in self.scatterer_ifaces:
            if not (1 <= idx <= M):
                msg = (
                    f"Scatterer interface {idx} out of range [1, {M}] "
                    f"for a model with {self.model.n_layers} layers"
                )
                raise ValueError(msg)
            if idx not in self.tmatrices:
                msg = f"Missing T-matrix for scatterer interface {idx}"
                raise ValueError(msg)
            T = self.tmatrices[idx]
            if T.shape != (9, 9):
                msg = (
                    f"T-matrix at interface {idx} has shape {T.shape}, expected (9, 9)"
                )
                raise ValueError(msg)
            if idx not in self.number_densities:
                msg = f"Missing number density for scatterer interface {idx}"
                raise ValueError(msg)


@dataclass
class InterlayerMSResult9x9:
    """Result of 9×9 interlayer multiple scattering calculation.

    Args:
        R_background: Background reflectivity (no scatterers), shape (n,).
        R_total: Total reflectivity with interlayer MS, shape (n,).
        R_born: Born (single-scattering) approximation, shape (n,).
        psi_exciting: Exciting field at each scatterer interface,
            dict mapping iface → (n, 9).
        psi_incident: Incident field at each scatterer interface,
            dict mapping iface → (n, 9).
    """

    R_background: np.ndarray
    R_total: np.ndarray
    R_born: np.ndarray
    psi_exciting: dict[int, np.ndarray]
    psi_incident: dict[int, np.ndarray]


# ---------------------------------------------------------------------------
# Background incident field
# ---------------------------------------------------------------------------


def background_incident_field(
    model: LayerModel,
    omega: complex,
    kH: np.ndarray,
    scatterer_ifaces: list[int],
) -> dict[int, np.ndarray]:
    """Compute the background incident field at each scatterer interface.

    The incident field is the response to a unit σ_zz source at the ocean
    bottom (interface 0), evaluated at each scatterer interface.  This
    corresponds to column 2 of G(j, 0) — the σ_zz source column.

    Args:
        model: Stratified elastic model.
        omega: Angular frequency (scalar, may be complex).
        kH: Horizontal wavenumber magnitudes, shape (n_kH,).
        scatterer_ifaces: Interface indices with scatterers.

    Returns:
        Dict mapping interface index → incident field, shape (n_kH, 4).
    """
    psi0: dict[int, np.ndarray] = {}
    for j in scatterer_ifaces:
        G_j0 = layered_greens_psv(model, omega, kH, source_iface=0, receiver_iface=j)
        # Column 2 = σ_zz source
        psi0[j] = G_j0[:, :, 2]
    return psi0


# ---------------------------------------------------------------------------
# Interlayer Green's matrix
# ---------------------------------------------------------------------------


def build_interlayer_greens_matrix(
    model: LayerModel,
    omega: complex,
    kH: np.ndarray,
    scatterer_ifaces: list[int],
) -> np.ndarray:
    """Build the interlayer Green's matrix for the Foldy-Lax system.

    Constructs a block matrix G_block of shape (n_kH, 4*N_z, 4*N_z) where
    N_z = len(scatterer_ifaces).  The (i, j) block is the 4×4 Green's
    function from scatterer interface j to scatterer interface i, with
    diagonal blocks set to zero (no intralayer scattering).

    Args:
        model: Stratified elastic model.
        omega: Angular frequency.
        kH: Horizontal wavenumber magnitudes, shape (n_kH,).
        scatterer_ifaces: Interface indices with scatterers.

    Returns:
        G_block: shape (n_kH, 4*N_z, 4*N_z).
    """
    n_kH = len(kH)
    N_z = len(scatterer_ifaces)
    G_block = np.zeros((n_kH, 4 * N_z, 4 * N_z), dtype=_CD)

    # Precompute model arrays once
    nlayer, E_d, E_u, phase_d, e_d_oc, e_u_oc, e0 = _prepare_model_arrays(
        model, omega, kH
    )

    for i_idx, iface_i in enumerate(scatterer_ifaces):
        for j_idx, iface_j in enumerate(scatterer_ifaces):
            if i_idx == j_idx:
                continue  # diagonal = 0 (no intralayer scattering)
            G_ij = riccati_greens_psv(
                nlayer,
                E_d,
                E_u,
                phase_d,
                e_d_oc,
                e_u_oc,
                e0,
                source_iface=iface_j,
                receiver_iface=iface_i,
            )
            r0, r1 = 4 * i_idx, 4 * (i_idx + 1)
            c0, c1 = 4 * j_idx, 4 * (j_idx + 1)
            G_block[:, r0:r1, c0:c1] = G_ij

    return G_block


# ---------------------------------------------------------------------------
# Foldy-Lax solve
# ---------------------------------------------------------------------------


def solve_interlayer_foldy_lax(
    G_block: np.ndarray,
    T_block_diag: np.ndarray,
    psi_incident: np.ndarray,
) -> np.ndarray:
    """Solve the Foldy-Lax system for exciting field amplitudes.

    Solves (I − G_block @ T_block_diag) ψ = ψ⁰ per wavenumber.

    Args:
        G_block: Interlayer Green's matrix, shape (n_kH, 4*N_z, 4*N_z).
        T_block_diag: Block-diagonal T-matrix (n_j * T_j per block),
            shape (n_kH, 4*N_z, 4*N_z).
        psi_incident: Stacked incident field, shape (n_kH, 4*N_z).

    Returns:
        psi_exciting: Stacked exciting field, shape (n_kH, 4*N_z).
    """
    n_kH = G_block.shape[0]
    N = G_block.shape[1]
    eye = np.eye(N, dtype=_CD)[np.newaxis, :, :]  # (1, N, N)
    A = eye - G_block @ T_block_diag  # (n_kH, N, N)
    return np.linalg.solve(A, psi_incident[..., np.newaxis])[..., 0]


# ---------------------------------------------------------------------------
# Scattered reflectivity
# ---------------------------------------------------------------------------


def _extract_ocean_upgoing(
    model: LayerModel,
    omega: complex,
    kH: np.ndarray,
    scattered_state: np.ndarray,
) -> np.ndarray:
    """Extract upgoing ocean P-wave amplitude from scattered state at iface 0.

    The scattered field at the ocean bottom has no incident ocean P-wave
    (D₀ᴾ = 0), so the 2×2 ocean system simplifies.

    State at ocean bottom: [u_x, u_z, σ_zz/(−iω), σ_xz/(−iω)].
    Ocean has u_x = 0 and σ_xz = 0. The 2 remaining equations:
        u_z = e_d_oc[0]*D₀ᴾ + e_u_oc[0]*U₀ᴾ
        σ_zz/(−iω) = e_d_oc[1]*D₀ᴾ + e_u_oc[1]*U₀ᴾ
    With D₀ᴾ = 0 for scattered field:
        U₀ᴾ = σ_zz_scat / e_u_oc[1]

    Args:
        model: Stratified elastic model.
        omega: Angular frequency.
        kH: Horizontal wavenumbers, shape (n_kH,).
        scattered_state: Scattered state at ocean bottom, shape (n_kH, 4).

    Returns:
        U0_P: Upgoing ocean P-wave amplitude, shape (n_kH,).
    """
    nlayer, E_d, E_u, phase_d, e_d_oc, e_u_oc, e0 = _prepare_model_arrays(
        model, omega, kH
    )
    # e_u_oc: shape (n_kH, 2), row 0 = u_z, row 1 = σ_zz/(−iω)
    # σ_zz/(−iω) component of scattered state is index 2
    sigma_zz_scat = scattered_state[:, 2]
    U0_P = sigma_zz_scat / e_u_oc[:, 1]
    return U0_P


def scattered_reflectivity(
    model: LayerModel,
    omega: complex,
    kH: np.ndarray,
    slab: "ScattererSlab",
    psi_exciting: dict[int, np.ndarray],
) -> np.ndarray:
    """Compute scattered reflectivity from exciting field amplitudes.

    For each scatterer interface j, the scattered source is n_j T_j ψ_j.
    The total scattered state at the ocean bottom is:
        Σ_j G(0, j) @ (n_j T_j @ ψ_j)

    The upgoing ocean P-wave amplitude U₀ᴾ is extracted and converted
    to reflectivity: R_scat = e0 * U₀ᴾ.

    Args:
        model: Stratified elastic model.
        omega: Angular frequency.
        kH: Horizontal wavenumbers, shape (n_kH,).
        slab: Scatterer configuration.
        psi_exciting: Dict mapping iface → exciting field (n_kH, 4).

    Returns:
        R_scattered: Scattered reflectivity perturbation, shape (n_kH,).
    """
    n_kH = len(kH)

    # Precompute model arrays for ocean extraction
    nlayer, E_d, E_u, phase_d, e_d_oc, e_u_oc, e0 = _prepare_model_arrays(
        model, omega, kH
    )

    # Accumulate scattered state at ocean bottom
    scattered_state = np.zeros((n_kH, 4), dtype=_CD)
    for j in slab.scatterer_ifaces:
        n_j = slab.number_densities[j]
        T_j = slab.tmatrices[j]
        psi_j = psi_exciting[j]  # (n_kH, 4)

        # Scattered source at interface j: n_j * T_j @ ψ_j
        source_j = n_j * (psi_j @ T_j.T)  # (n_kH, 4)

        # G(0, j) @ source_j
        G_0j = riccati_greens_psv(
            nlayer,
            E_d,
            E_u,
            phase_d,
            e_d_oc,
            e_u_oc,
            e0,
            source_iface=j,
            receiver_iface=0,
        )
        # G_0j: (n_kH, 4, 4), source_j: (n_kH, 4)
        scattered_state += np.einsum("fij,fj->fi", G_0j, source_j)

    # Extract upgoing ocean P-wave amplitude
    U0_P = scattered_state[:, 2] / e_u_oc[:, 1]
    R_scattered = e0 * U0_P

    return R_scattered


# ---------------------------------------------------------------------------
# Background reflectivity
# ---------------------------------------------------------------------------


def _background_reflectivity(
    model: LayerModel,
    omega: complex,
    kH: np.ndarray,
) -> np.ndarray:
    """Compute background ocean-bottom reflectivity (no scatterers).

    Uses G(0, 0) column 2 to extract R from the ocean P-wave source.

    Args:
        model: Stratified elastic model.
        omega: Angular frequency.
        kH: Horizontal wavenumbers, shape (n_kH,).

    Returns:
        R_background: shape (n_kH,).
    """
    nlayer, E_d, E_u, phase_d, e_d_oc, e_u_oc, e0 = _prepare_model_arrays(
        model, omega, kH
    )
    G_00 = riccati_greens_psv(
        nlayer,
        E_d,
        E_u,
        phase_d,
        e_d_oc,
        e_u_oc,
        e0,
        source_iface=0,
        receiver_iface=0,
    )
    # From G(0,0) column 2 (σ_zz source), extract u_z (row 1)
    # Ocean system: u_z = e_d_oc[0]*D₀ᴾ*e0 + e_u_oc[0]*U₀ᴾ
    #               σ_zz/(−iω) = e_d_oc[1]*D₀ᴾ*e0 + e_u_oc[1]*U₀ᴾ
    # The Green's function at iface 0 with source at iface 0 gives
    # the state at the ocean bottom. We need to decompose into
    # incident (D₀ᴾ) and scattered (U₀ᴾ) parts.
    # For background R: use σ_zz column, and solve the 2x2 ocean system.
    # state = G_00[:, :, 2]  →  [u_x, u_z, σ_zz/(−iω), σ_xz/(−iω)]
    state = G_00[:, :, 2]  # (n_kH, 4)
    u_z = state[:, 1]
    sigma_zz = state[:, 2]

    # Solve: [e_d_oc[0]*e0, e_u_oc[0]] [D₀ᴾ]   [u_z      ]
    #        [e_d_oc[1]*e0, e_u_oc[1]] [U₀ᴾ] = [σ_zz/(−iω)]
    det = e_d_oc[:, 0] * e0 * e_u_oc[:, 1] - e_u_oc[:, 0] * e_d_oc[:, 1] * e0
    U0_P = (e_d_oc[:, 0] * e0 * sigma_zz - e_d_oc[:, 1] * e0 * u_z) / det
    R_background = e0 * U0_P

    return R_background


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------


def interlayer_ms_reflectivity(
    slab: ScattererSlab,
    omega: complex,
    kH: np.ndarray,
) -> InterlayerMSResult:
    """Compute reflectivity with interlayer multiple scattering.

    Top-level convenience function that chains:
        background → Green's matrix → Foldy-Lax → reflectivity.

    Args:
        slab: Scatterer configuration.
        omega: Angular frequency (scalar, may be complex).
        kH: Horizontal wavenumber magnitudes, shape (n_kH,).

    Returns:
        InterlayerMSResult with background, total, and Born reflectivities.
    """
    model = slab.model
    ifaces = slab.scatterer_ifaces
    N_z = len(ifaces)
    n_kH = len(kH)

    # 1. Background reflectivity
    R_bg = _background_reflectivity(model, omega, kH)

    # 2. Incident field at scatterer interfaces
    psi0 = background_incident_field(model, omega, kH, ifaces)

    # 3. Build interlayer Green's matrix
    G_block = build_interlayer_greens_matrix(model, omega, kH, ifaces)

    # 4. Build block-diagonal T-matrix: T_block = blkdiag(n_j * T_j)
    T_block = np.zeros((n_kH, 4 * N_z, 4 * N_z), dtype=_CD)
    for idx, j in enumerate(ifaces):
        n_j = slab.number_densities[j]
        T_j = slab.tmatrices[j]
        r0, r1 = 4 * idx, 4 * (idx + 1)
        T_block[:, r0:r1, r0:r1] = n_j * T_j[np.newaxis, :, :]

    # 5. Stack incident field
    psi_inc_stacked = np.zeros((n_kH, 4 * N_z), dtype=_CD)
    for idx, j in enumerate(ifaces):
        psi_inc_stacked[:, 4 * idx : 4 * (idx + 1)] = psi0[j]

    # 6. Foldy-Lax solve
    psi_exc_stacked = solve_interlayer_foldy_lax(G_block, T_block, psi_inc_stacked)

    # 7. Unstack exciting field
    psi_exciting: dict[int, np.ndarray] = {}
    for idx, j in enumerate(ifaces):
        psi_exciting[j] = psi_exc_stacked[:, 4 * idx : 4 * (idx + 1)]

    # 8. Scattered reflectivity (Foldy-Lax)
    R_scat = scattered_reflectivity(model, omega, kH, slab, psi_exciting)
    R_total = R_bg + R_scat

    # 9. Born approximation: use incident field instead of exciting field
    R_born_scat = scattered_reflectivity(model, omega, kH, slab, psi0)
    R_born = R_bg + R_born_scat

    return InterlayerMSResult(
        R_background=R_bg,
        R_total=R_total,
        R_born=R_born,
        psi_exciting=psi_exciting,
        psi_incident=psi0,
    )


# ---------------------------------------------------------------------------
# T-matrix conversion helpers
# ---------------------------------------------------------------------------


def tmatrix_6x6_to_4x4_psv(T_6x6: np.ndarray) -> np.ndarray:
    """Extract 4×4 P-SV block from a 6×6 displacement-traction T-matrix.

    The 6×6 ordering is (u_z, u_x, u_y, σ_zz, σ_xz, σ_yz).
    The P-SV state vector is [u_x, u_z, σ_zz/(−iω), σ_xz/(−iω)].

    We extract indices [1, 0, 3, 4] (u_x, u_z, σ_zz, σ_xz) from both
    rows and columns, noting that the stress normalisation by (−iω) must
    be handled by the caller when the T-matrix operates in physical units.

    Args:
        T_6x6: T-matrix in (u_z, u_x, u_y, σ_zz, σ_xz, σ_yz) basis,
            shape (6, 6).

    Returns:
        T_4x4: P-SV T-matrix in (u_x, u_z, σ_zz, σ_xz) basis,
            shape (4, 4).
    """
    idx = [1, 0, 3, 4]
    return T_6x6[np.ix_(idx, idx)]


def tmatrix_9x9_to_4x4_psv(
    T_9x9: np.ndarray,
    kx: float,
    ky: float,
    rho: float,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Convert a 9×9 (u, ε) T-matrix to 4×4 P-SV (u, t) T-matrix.

    Conversion chain: ``T_6x6 = B @ T_9x9 @ A`` then extract P-SV.

    Args:
        T_9x9: T-matrix in (u, ε) basis, shape (9, 9).
        kx: Horizontal wavenumber x-component.
        ky: Horizontal wavenumber y-component.
        rho: Layer density at scatterer interface.
        alpha: P-wave velocity at scatterer interface.
        beta: S-wave velocity at scatterer interface.

    Returns:
        T_4x4: P-SV T-matrix, shape (4, 4).
    """
    A = strain_from_displacement_traction(
        np.atleast_1d(kx), np.atleast_1d(ky), rho, alpha, beta
    )
    B = traction_from_strain(rho, alpha, beta)
    T_6x6 = B @ T_9x9 @ A.squeeze()
    return tmatrix_6x6_to_4x4_psv(T_6x6)


# ---------------------------------------------------------------------------
# 9×9 background incident field
# ---------------------------------------------------------------------------


def background_incident_field_9x9(
    model: LayerModel,
    omega: complex,
    kx: np.ndarray,
    ky: np.ndarray,
    scatterer_ifaces: list[int],
) -> dict[int, np.ndarray]:
    """Background incident field in (u, ε) basis at each scatterer interface.

    The incident field is ``A_j @ G_6x6(j, 0)[:, :, 3]`` — the
    displacement-strain response to a unit physical σ_zz source at the
    ocean bottom.

    Args:
        model: Stratified elastic model.
        omega: Angular frequency (scalar, may be complex).
        kx: Horizontal wavenumber x-component, shape (n,).
        ky: Horizontal wavenumber y-component, shape (n,).
        scatterer_ifaces: Interface indices with scatterers.

    Returns:
        Dict mapping interface index → incident field, shape (n, 9).
    """
    psi0: dict[int, np.ndarray] = {}
    for j in scatterer_ifaces:
        G6_j0 = layered_greens_6x6(
            model, omega, kx, ky, source_iface=0, receiver_iface=j
        )
        rho_j, alpha_j, beta_j = _interface_elastic_properties(model, j)
        A_j = strain_from_displacement_traction(kx, ky, rho_j, alpha_j, beta_j)
        # Column 3 = σ_zz source in the 6×6 physical basis
        psi0[j] = np.einsum("...ij,...j->...i", A_j, G6_j0[..., :, 3])
    return psi0


# ---------------------------------------------------------------------------
# 9×9 interlayer Green's matrix
# ---------------------------------------------------------------------------


def build_interlayer_greens_matrix_9x9(
    model: LayerModel,
    omega: complex,
    kx: np.ndarray,
    ky: np.ndarray,
    scatterer_ifaces: list[int],
) -> np.ndarray:
    """Build interlayer Green's matrix in the 9×9 (u, ε) basis.

    Constructs a block matrix of shape ``(n, 9*N_z, 9*N_z)`` where
    ``N_z = len(scatterer_ifaces)``.  Diagonal blocks are zero.

    Args:
        model: Stratified elastic model.
        omega: Angular frequency.
        kx: Horizontal wavenumber x-component, shape (n,).
        ky: Horizontal wavenumber y-component, shape (n,).
        scatterer_ifaces: Interface indices with scatterers.

    Returns:
        G_block: shape ``(n, 9*N_z, 9*N_z)``.
    """
    n = len(kx)
    N_z = len(scatterer_ifaces)
    G_block = np.zeros((n, 9 * N_z, 9 * N_z), dtype=_CD)

    for i_idx, iface_i in enumerate(scatterer_ifaces):
        for j_idx, iface_j in enumerate(scatterer_ifaces):
            if i_idx == j_idx:
                continue
            G9_ij = layered_greens_9x9(
                model,
                omega,
                kx,
                ky,
                source_iface=iface_j,
                receiver_iface=iface_i,
            )
            r0, r1 = 9 * i_idx, 9 * (i_idx + 1)
            c0, c1 = 9 * j_idx, 9 * (j_idx + 1)
            G_block[:, r0:r1, c0:c1] = G9_ij

    return G_block


# ---------------------------------------------------------------------------
# 9×9 scattered reflectivity
# ---------------------------------------------------------------------------


def scattered_reflectivity_9x9(
    model: LayerModel,
    omega: complex,
    kx: np.ndarray,
    ky: np.ndarray,
    slab: ScattererSlab9x9,
    psi_exciting: dict[int, np.ndarray],
) -> np.ndarray:
    """Scattered reflectivity from 9×9 exciting field amplitudes.

    Converts each scatterer's scattered (u, ε) source to (u, t) via B,
    propagates to the ocean bottom via ``G_6x6(0, j)``, and extracts the
    upgoing ocean P-wave amplitude.

    Args:
        model: Stratified elastic model.
        omega: Angular frequency.
        kx: Horizontal wavenumber x-component, shape (n,).
        ky: Horizontal wavenumber y-component, shape (n,).
        slab: 9×9 scatterer configuration.
        psi_exciting: Dict mapping iface → exciting field (n, 9).

    Returns:
        R_scattered: Scattered reflectivity perturbation, shape (n,).
    """
    n = len(kx)
    kH = np.sqrt(kx**2 + ky**2)
    kH_safe = np.where(kH > 0, kH, 1e-30)

    nlayer, E_d, E_u, phase_d, e_d_oc, e_u_oc, e0 = _prepare_model_arrays(
        model, omega, kH_safe
    )

    # Accumulate scattered state at ocean bottom in 6×6 physical basis
    scattered_state_6 = np.zeros((n, 6), dtype=_CD)
    for j in slab.scatterer_ifaces:
        n_j = slab.number_densities[j]
        T_j = slab.tmatrices[j]  # (9, 9)
        psi_j = psi_exciting[j]  # (n, 9)

        # Scattered source in (u, ε) basis
        source_9 = n_j * (psi_j @ T_j.T)  # (n, 9)

        # Convert to physical (u, t) via B
        rho_j, alpha_j, beta_j = _interface_elastic_properties(model, j)
        B_j = traction_from_strain(rho_j, alpha_j, beta_j)  # (6, 9)
        source_6 = source_9 @ B_j.T  # (n, 6)

        # Propagate via G_6x6(0, j)
        G6_0j = layered_greens_6x6(
            model, omega, kx, ky, source_iface=j, receiver_iface=0
        )
        scattered_state_6 += np.einsum("...ij,...j->...i", G6_0j, source_6)

    # Extract upgoing ocean P-wave from physical 6×6 state.
    # Physical σ_zz is at index 3; convert to Riccati convention.
    miw = -1j * omega
    sigma_zz_riccati = scattered_state_6[:, 3] / miw
    U0_P = sigma_zz_riccati / e_u_oc[:, 1]
    return e0 * U0_P


# ---------------------------------------------------------------------------
# 9×9 top-level convenience function
# ---------------------------------------------------------------------------


def interlayer_ms_reflectivity_9x9(
    slab: ScattererSlab9x9,
    omega: complex,
    kx: np.ndarray,
    ky: np.ndarray,
) -> InterlayerMSResult9x9:
    """Reflectivity with interlayer multiple scattering (9×9 basis).

    Top-level convenience function: background → 9×9 Green's → Foldy-Lax
    → reflectivity.

    Args:
        slab: 9×9 scatterer configuration.
        omega: Angular frequency (scalar, may be complex).
        kx: Horizontal wavenumber x-component, shape (n,).
        ky: Horizontal wavenumber y-component, shape (n,).

    Returns:
        InterlayerMSResult9x9 with background, total, and Born reflectivities.
    """
    model = slab.model
    ifaces = slab.scatterer_ifaces
    N_z = len(ifaces)
    n = len(kx)

    # 1. Background reflectivity (reuse 4×4 path)
    kH = np.sqrt(kx**2 + ky**2)
    kH_safe = np.where(kH > 0, kH, 1e-30)
    R_bg = _background_reflectivity(model, omega, kH_safe)

    # 2. Incident field in 9×9 basis
    psi0 = background_incident_field_9x9(model, omega, kx, ky, ifaces)

    # 3. Build interlayer Green's matrix in 9×9 basis
    G_block = build_interlayer_greens_matrix_9x9(model, omega, kx, ky, ifaces)

    # 4. Block-diagonal T-matrix: T_block = blkdiag(n_j * T_j)
    T_block = np.zeros((n, 9 * N_z, 9 * N_z), dtype=_CD)
    for idx, j in enumerate(ifaces):
        n_j = slab.number_densities[j]
        T_j = slab.tmatrices[j]
        r0, r1 = 9 * idx, 9 * (idx + 1)
        T_block[:, r0:r1, r0:r1] = n_j * T_j[np.newaxis, :, :]

    # 5. Stack incident field
    psi_inc_stacked = np.zeros((n, 9 * N_z), dtype=_CD)
    for idx, j in enumerate(ifaces):
        psi_inc_stacked[:, 9 * idx : 9 * (idx + 1)] = psi0[j]

    # 6. Foldy-Lax solve (reuse generic solver)
    psi_exc_stacked = solve_interlayer_foldy_lax(G_block, T_block, psi_inc_stacked)

    # 7. Unstack exciting field
    psi_exciting: dict[int, np.ndarray] = {}
    for idx, j in enumerate(ifaces):
        psi_exciting[j] = psi_exc_stacked[:, 9 * idx : 9 * (idx + 1)]

    # 8. Scattered reflectivity (Foldy-Lax)
    R_scat = scattered_reflectivity_9x9(model, omega, kx, ky, slab, psi_exciting)
    R_total = R_bg + R_scat

    # 9. Born approximation: incident field instead of exciting field
    R_born_scat = scattered_reflectivity_9x9(model, omega, kx, ky, slab, psi0)
    R_born = R_bg + R_born_scat

    return InterlayerMSResult9x9(
        R_background=R_bg,
        R_total=R_total,
        R_born=R_born,
        psi_exciting=psi_exciting,
        psi_incident=psi0,
    )
