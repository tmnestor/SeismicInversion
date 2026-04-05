"""Global Matrix Method for plane-wave reflectivity (NumPy).

Assembles all interface continuity conditions into a single dense linear system
G x = b and solves for the wave amplitudes. This is mathematically equivalent to
Kennett's recursive method but yields derivatives more cheaply: factor G once
then back-substitute for each parameter perturbation.

Reference: Chin, Hedstrom & Thigpen (1984).
"""

import numpy as np

from Kennett_Reflectivity.layer_model import (
    LayerModel,
    vertical_slowness,
)

from .layer_matrix import layer_eigenvectors, ocean_eigenvectors

__all__ = ["gmm_reflectivity"]


def _build_system(
    nlayer: int,
    eta: np.ndarray,
    neta: np.ndarray,
    rho: np.ndarray,
    beta_c: np.ndarray,
    thickness: np.ndarray,
    p: complex,
    omega: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Assemble the global matrix system G x = b for all frequencies.

    Returns:
        G: shape (nfreq, N, N)
        b: shape (nfreq, N)
        N: system size
    """
    nfreq = len(omega)

    # Determine system size: ocean(1) + elastic_finite(4 each) + half-space(2)
    # Ocean contributes 1 unknown (U0^P; D0^P = 1 is the source)
    # Each finite elastic layer: 4 unknowns (D^P, D^S, U^P, U^S)
    # Half-space: 2 unknowns (D^P, D^S)
    n_elastic_finite = nlayer - 2  # layers 1..nlayer-2
    N = 1 + 4 * n_elastic_finite + 2

    G = np.zeros((nfreq, N, N), dtype=np.complex128)
    b = np.zeros((nfreq, N), dtype=np.complex128)

    # Precompute E-matrices for all solid layers (frequency-independent)
    E_d = {}
    E_u = {}
    for j in range(1, nlayer):
        E_d[j], E_u[j] = layer_eigenvectors(p, eta[j], neta[j], rho[j], beta_c[j])

    # Phase factors: exp(i * omega * eta_j * h_j) for finite layers
    # P-wave phase through ocean
    e0 = np.exp(1j * omega * eta[0] * thickness[0])  # (nfreq,)

    # Phase diagonals for solid finite layers
    phase_d = {}  # phase_d[j] = (nfreq, 2) diagonal [exp(iw*eta*h), exp(iw*neta*h)]
    for j in range(1, nlayer - 1):
        ph_p = np.exp(1j * omega * eta[j] * thickness[j])
        ph_s = np.exp(1j * omega * neta[j] * thickness[j])
        phase_d[j] = np.stack([ph_p, ph_s], axis=-1)  # (nfreq, 2)

    # Column offset mapping:
    # col 0: U0^P (ocean upgoing)
    # cols 1..4: layer 1 (D^P, D^S, U^P, U^S)
    # cols 5..8: layer 2
    # ...
    # last 2 cols: half-space (D^P, D^S)
    def col_offset(layer_idx: int) -> int:
        if layer_idx == 0:
            return 0  # ocean: 1 unknown at col 0
        if layer_idx < nlayer - 1:
            return 1 + 4 * (layer_idx - 1)  # elastic finite
        return 1 + 4 * n_elastic_finite  # half-space

    row = 0

    # ===== Interface 0: Ocean-bottom (ocean / layer 1) =====
    # 3 equations: u_z continuity, sigma_zz continuity, sigma_xz = 0
    #
    # Ocean side: D0^P = 1 (source), U0^P is unknown
    # e_d_oc, e_u_oc are the ocean eigenvectors (2-component: u_z, sigma_zz)
    # Solid side (top of layer 1): E_d[1] @ D1 + E_u[1] @ P1 @ U1
    # where P1 = diag(phase) applies to upgoing amplitudes at bottom

    e_d_oc, e_u_oc = ocean_eigenvectors(p, eta[0], rho[0])

    # Rows from interface 0:
    # Row 0 (u_z): e_u_oc[0] * U0^P - (E1_top)[row1, :] @ x1 = -e_d_oc[0] * e0
    #   where E1_top = [E_d[1] | E_u[1] * P1] and x1 = [D1^P, D1^S, U1^P, U1^S]
    # Row 1 (sigma_zz): e_u_oc[1] * U0^P - (E1_top)[row2, :] @ x1 = -e_d_oc[1] * e0
    #   Note: sigma_zz is -rho for ocean, matching row 2 of E-matrix
    # Row 2 (sigma_xz = 0): 0 = (E1_top)[row3, :] @ x1

    # But we must be careful: the ocean eigenvectors give [u_z, sigma_zz/(-iw)]
    # with the convention that downgoing has e_d = [eta, -rho] and upgoing e_u = [-eta, -rho].
    # The phase for the downgoing wave through the ocean is e0 = exp(i*w*eta0*h0).
    # The upgoing wave U0^P is referenced at the ocean bottom (no phase needed here,
    # since we're evaluating at the bottom).

    # For the solid layer 1:
    # Downgoing D referenced at top: no phase at top
    # Upgoing U referenced at bottom: multiply by P at top → E_u * diag(phase)

    c_ocean = 0  # column for U0^P
    c_layer1 = col_offset(1)  # columns for layer 1

    if nlayer > 2:
        # Layer 1 is a finite elastic layer with 4 unknowns
        # E1_top columns: [E_d[1][:,0], E_d[1][:,1], E_u[1][:,0]*P1_p, E_u[1][:,1]*P1_s]
        # We need rows 1,2,3 of the E-matrix (u_z, sigma_zz, sigma_xz)

        # u_z continuity (E-matrix row 1)
        # ocean side: e_u_oc[0] * U0^P
        G[:, row, c_ocean] = e_u_oc[0]
        # solid side: -E_d[1][1,:] @ [D^P, D^S] - E_u[1][1,:]*P @ [U^P, U^S]
        G[:, row, c_layer1] = -E_d[1][1, 0]
        G[:, row, c_layer1 + 1] = -E_d[1][1, 1]
        G[:, row, c_layer1 + 2] = -E_u[1][1, 0] * phase_d[1][:, 0]
        G[:, row, c_layer1 + 3] = -E_u[1][1, 1] * phase_d[1][:, 1]
        # RHS: -e_d_oc[0] * e0  (from D0^P = 1, phase through ocean)
        b[:, row] = -e_d_oc[0] * e0
        row += 1

        # sigma_zz continuity (E-matrix row 2)
        G[:, row, c_ocean] = e_u_oc[1]
        G[:, row, c_layer1] = -E_d[1][2, 0]
        G[:, row, c_layer1 + 1] = -E_d[1][2, 1]
        G[:, row, c_layer1 + 2] = -E_u[1][2, 0] * phase_d[1][:, 0]
        G[:, row, c_layer1 + 3] = -E_u[1][2, 1] * phase_d[1][:, 1]
        b[:, row] = -e_d_oc[1] * e0
        row += 1

        # sigma_xz = 0 at ocean bottom (E-matrix row 3)
        G[:, row, c_layer1] = -E_d[1][3, 0]
        G[:, row, c_layer1 + 1] = -E_d[1][3, 1]
        G[:, row, c_layer1 + 2] = -E_u[1][3, 0] * phase_d[1][:, 0]
        G[:, row, c_layer1 + 3] = -E_u[1][3, 1] * phase_d[1][:, 1]
        row += 1
    else:
        # Layer 1 is the half-space (only 2 unknowns: D^P, D^S, no upgoing)
        G[:, row, c_ocean] = e_u_oc[0]
        G[:, row, c_layer1] = -E_d[1][1, 0]
        G[:, row, c_layer1 + 1] = -E_d[1][1, 1]
        b[:, row] = -e_d_oc[0] * e0
        row += 1

        G[:, row, c_ocean] = e_u_oc[1]
        G[:, row, c_layer1] = -E_d[1][2, 0]
        G[:, row, c_layer1 + 1] = -E_d[1][2, 1]
        b[:, row] = -e_d_oc[1] * e0
        row += 1

        G[:, row, c_layer1] = -E_d[1][3, 0]
        G[:, row, c_layer1 + 1] = -E_d[1][3, 1]
        row += 1

    # ===== Solid-solid interfaces (between layers k and k+1) =====
    for k in range(1, nlayer - 1):
        k_next = k + 1
        c_k = col_offset(k)
        c_next = col_offset(k_next)

        # At bottom of layer k / top of layer k+1:
        # Layer k bottom: E_d[k]*P_k @ D_k + E_u[k] @ U_k
        # Layer k+1 top: E_d[k+1] @ D_{k+1} + E_u[k+1]*P_{k+1} @ U_{k+1}
        # (if k+1 is half-space, no upgoing and no phase)

        # 4 equations (all 4 rows of E-matrix must match)
        for eq_row in range(4):
            # Layer k side (left of interface)
            # Downgoing referenced at top, evaluated at bottom → multiply by phase
            G[:, row, c_k] += E_d[k][eq_row, 0] * phase_d[k][:, 0]
            G[:, row, c_k + 1] += E_d[k][eq_row, 1] * phase_d[k][:, 1]
            # Upgoing referenced at bottom, no phase needed at bottom
            G[:, row, c_k + 2] += E_u[k][eq_row, 0]
            G[:, row, c_k + 3] += E_u[k][eq_row, 1]

            # Layer k+1 side (right of interface) — subtract
            if k_next < nlayer - 1:
                # Finite elastic layer: 4 unknowns
                G[:, row, c_next] -= E_d[k_next][eq_row, 0]
                G[:, row, c_next + 1] -= E_d[k_next][eq_row, 1]
                G[:, row, c_next + 2] -= E_u[k_next][eq_row, 0] * phase_d[k_next][:, 0]
                G[:, row, c_next + 3] -= E_u[k_next][eq_row, 1] * phase_d[k_next][:, 1]
            else:
                # Half-space: only downgoing, no phase
                G[:, row, c_next] -= E_d[k_next][eq_row, 0]
                G[:, row, c_next + 1] -= E_d[k_next][eq_row, 1]

            row += 1

    return G, b, N


def gmm_reflectivity(
    model: LayerModel,
    p: float,
    omega: np.ndarray,
    free_surface: bool = False,
) -> np.ndarray:
    """Compute plane-wave reflectivity using the Global Matrix Method.

    Args:
        model: Stratified elastic model.
        p: Horizontal slowness (ray parameter).
        omega: Angular frequencies, shape (nfreq,). Must not include DC.
        free_surface: Include free-surface reverberations.

    Returns:
        Complex PP reflectivity at each frequency, shape (nfreq,).
    """
    nlayer = model.n_layers

    # Complex slownesses
    s_p = model.complex_slowness_p()
    s_s = model.complex_slowness_s()
    beta_c = model.complex_velocity_s()

    # Vertical slownesses
    cp = complex(p)
    eta = np.array(
        [vertical_slowness(s_p[i], cp) for i in range(nlayer)], dtype=np.complex128
    )
    neta = np.zeros(nlayer, dtype=np.complex128)
    for i in range(1, nlayer):
        neta[i] = vertical_slowness(s_s[i], cp)

    # Assemble and solve
    G, b_vec, N = _build_system(
        nlayer, eta, neta, model.rho, beta_c, model.thickness, cp, omega
    )

    x = np.linalg.solve(G, b_vec[..., np.newaxis])[..., 0]  # (nfreq, N)

    # U0^P is x[:, 0]. The reflectivity is the upgoing amplitude at the surface:
    # R = exp(i*omega*eta0*h0) * U0^P
    e0 = np.exp(1j * omega * eta[0] * model.thickness[0])
    R = e0 * x[:, 0]

    if free_surface:
        R = R / (1.0 + R)

    return R
