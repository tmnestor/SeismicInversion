"""Global Matrix Method — differentiable PyTorch implementation.

Mirrors ``global_matrix.py`` but uses ``torch`` operations so that
``torch.autograd`` can compute Jacobian and Hessian through the linear solve.

Convention: exp(-iwt) inverse Fourier transform, depth positive downward.
(Note: Chin, Hedstrom & Thigpen (1984) use the conjugate convention exp(+iwt).
All formulas here are adapted to match the Kennett implementation's convention.)
"""

import torch

from Kennett_Reflectivity.kennett_torch import (
    _pack_params,
    _unpack_params,
    complex_slowness_torch,
    vertical_slowness_torch,
)

from .layer_matrix import layer_eigenvectors_torch, ocean_eigenvectors_torch
from .riccati_solver import riccati_sweep_torch

__all__ = [
    "gmm_reflectivity_torch",
    "gmm_jacobian",
    "gmm_hessian",
]

_CDTYPE = torch.complex128
_FDTYPE = torch.float64


def _compute_eigenvectors_torch(
    nlayer: int,
    eta: torch.Tensor,
    neta: torch.Tensor,
    rho: torch.Tensor,
    beta_c: torch.Tensor,
    thickness: torch.Tensor,
    p: torch.Tensor,
    omega: torch.Tensor,
) -> tuple[
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Precompute eigenvectors and phase factors (PyTorch)."""
    cp = p.to(_CDTYPE)

    E_d: dict[int, torch.Tensor] = {}
    E_u: dict[int, torch.Tensor] = {}
    for j in range(1, nlayer):
        E_d[j], E_u[j] = layer_eigenvectors_torch(
            cp, eta[j], neta[j], rho[j], beta_c[j]
        )

    e0 = torch.exp(1j * omega * eta[0] * thickness[0])

    phase_d: dict[int, torch.Tensor] = {}
    for j in range(1, nlayer - 1):
        ph_p = torch.exp(1j * omega * eta[j] * thickness[j])
        ph_s = torch.exp(1j * omega * neta[j] * thickness[j])
        phase_d[j] = torch.stack([ph_p, ph_s], dim=-1)

    e_d_oc, e_u_oc = ocean_eigenvectors_torch(cp, eta[0], rho[0])

    return E_d, E_u, phase_d, e_d_oc, e_u_oc, e0


def _build_system_torch(
    nlayer: int,
    eta: torch.Tensor,
    neta: torch.Tensor,
    rho: torch.Tensor,
    beta_c: torch.Tensor,
    thickness: torch.Tensor,
    p: torch.Tensor,
    omega: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Assemble the global matrix system G x = b (PyTorch).

    Returns:
        G: shape (nfreq, N, N)
        b: shape (nfreq, N)
        N: system size
    """
    nfreq = omega.shape[0]
    cp = p.to(_CDTYPE)

    n_elastic_finite = nlayer - 2
    N = 1 + 4 * n_elastic_finite + 2

    # Build E-matrices for all solid layers
    E_d_list: dict[int, torch.Tensor] = {}
    E_u_list: dict[int, torch.Tensor] = {}
    for j in range(1, nlayer):
        E_d_list[j], E_u_list[j] = layer_eigenvectors_torch(
            cp, eta[j], neta[j], rho[j], beta_c[j]
        )

    # Phase through ocean
    e0 = torch.exp(1j * omega * eta[0] * thickness[0])  # (nfreq,)

    # Phase diagonals for finite solid layers
    phase_d: dict[int, torch.Tensor] = {}
    for j in range(1, nlayer - 1):
        ph_p = torch.exp(1j * omega * eta[j] * thickness[j])
        ph_s = torch.exp(1j * omega * neta[j] * thickness[j])
        phase_d[j] = torch.stack([ph_p, ph_s], dim=-1)  # (nfreq, 2)

    # We build G as a list of (row, col, values) and scatter into a dense tensor.
    # This avoids in-place ops that break autograd.
    # Strategy: build each row as a dense (nfreq, N) tensor, then stack.

    rows = []
    b_rows = []

    def col_offset(layer_idx: int) -> int:
        if layer_idx == 0:
            return 0
        if layer_idx < nlayer - 1:
            return 1 + 4 * (layer_idx - 1)
        return 1 + 4 * n_elastic_finite

    zero_f = torch.zeros(nfreq, dtype=_CDTYPE)

    def _make_row() -> list[torch.Tensor]:
        return [zero_f.clone() for _ in range(N)]

    # ===== Ocean-bottom interface =====
    e_d_oc, e_u_oc = ocean_eigenvectors_torch(cp, eta[0], rho[0])
    c_ocean = 0
    c_layer1 = col_offset(1)

    if nlayer > 2:
        # u_z continuity
        r = _make_row()
        r[c_ocean] = e_u_oc[0].expand(nfreq)
        r[c_layer1] = (-E_d_list[1][1, 0]).expand(nfreq)
        r[c_layer1 + 1] = (-E_d_list[1][1, 1]).expand(nfreq)
        r[c_layer1 + 2] = -E_u_list[1][1, 0] * phase_d[1][:, 0]
        r[c_layer1 + 3] = -E_u_list[1][1, 1] * phase_d[1][:, 1]
        rows.append(torch.stack(r, dim=-1))
        b_rows.append(-e_d_oc[0] * e0)

        # sigma_zz continuity
        r = _make_row()
        r[c_ocean] = e_u_oc[1].expand(nfreq)
        r[c_layer1] = (-E_d_list[1][2, 0]).expand(nfreq)
        r[c_layer1 + 1] = (-E_d_list[1][2, 1]).expand(nfreq)
        r[c_layer1 + 2] = -E_u_list[1][2, 0] * phase_d[1][:, 0]
        r[c_layer1 + 3] = -E_u_list[1][2, 1] * phase_d[1][:, 1]
        rows.append(torch.stack(r, dim=-1))
        b_rows.append(-e_d_oc[1] * e0)

        # sigma_xz = 0
        r = _make_row()
        r[c_layer1] = (-E_d_list[1][3, 0]).expand(nfreq)
        r[c_layer1 + 1] = (-E_d_list[1][3, 1]).expand(nfreq)
        r[c_layer1 + 2] = -E_u_list[1][3, 0] * phase_d[1][:, 0]
        r[c_layer1 + 3] = -E_u_list[1][3, 1] * phase_d[1][:, 1]
        rows.append(torch.stack(r, dim=-1))
        b_rows.append(zero_f.clone())
    else:
        # 2-layer model: layer 1 is half-space
        r = _make_row()
        r[c_ocean] = e_u_oc[0].expand(nfreq)
        r[c_layer1] = (-E_d_list[1][1, 0]).expand(nfreq)
        r[c_layer1 + 1] = (-E_d_list[1][1, 1]).expand(nfreq)
        rows.append(torch.stack(r, dim=-1))
        b_rows.append(-e_d_oc[0] * e0)

        r = _make_row()
        r[c_ocean] = e_u_oc[1].expand(nfreq)
        r[c_layer1] = (-E_d_list[1][2, 0]).expand(nfreq)
        r[c_layer1 + 1] = (-E_d_list[1][2, 1]).expand(nfreq)
        rows.append(torch.stack(r, dim=-1))
        b_rows.append(-e_d_oc[1] * e0)

        r = _make_row()
        r[c_layer1] = (-E_d_list[1][3, 0]).expand(nfreq)
        r[c_layer1 + 1] = (-E_d_list[1][3, 1]).expand(nfreq)
        rows.append(torch.stack(r, dim=-1))
        b_rows.append(zero_f.clone())

    # ===== Solid-solid interfaces =====
    for k in range(1, nlayer - 1):
        k_next = k + 1
        c_k = col_offset(k)
        c_next = col_offset(k_next)

        for eq_row in range(4):
            r = _make_row()
            # Layer k bottom
            r[c_k] = r[c_k] + E_d_list[k][eq_row, 0] * phase_d[k][:, 0]
            r[c_k + 1] = r[c_k + 1] + E_d_list[k][eq_row, 1] * phase_d[k][:, 1]
            r[c_k + 2] = r[c_k + 2] + E_u_list[k][eq_row, 0].expand(nfreq)
            r[c_k + 3] = r[c_k + 3] + E_u_list[k][eq_row, 1].expand(nfreq)

            # Layer k+1 top
            if k_next < nlayer - 1:
                r[c_next] = r[c_next] - E_d_list[k_next][eq_row, 0].expand(nfreq)
                r[c_next + 1] = r[c_next + 1] - E_d_list[k_next][eq_row, 1].expand(
                    nfreq
                )
                r[c_next + 2] = (
                    r[c_next + 2] - E_u_list[k_next][eq_row, 0] * phase_d[k_next][:, 0]
                )
                r[c_next + 3] = (
                    r[c_next + 3] - E_u_list[k_next][eq_row, 1] * phase_d[k_next][:, 1]
                )
            else:
                r[c_next] = r[c_next] - E_d_list[k_next][eq_row, 0].expand(nfreq)
                r[c_next + 1] = r[c_next + 1] - E_d_list[k_next][eq_row, 1].expand(
                    nfreq
                )

            rows.append(torch.stack(r, dim=-1))
            b_rows.append(zero_f.clone())

    G = torch.stack(rows, dim=1)  # (nfreq, N, N)
    b_vec = torch.stack(b_rows, dim=1)  # (nfreq, N)
    return G, b_vec, N


def gmm_reflectivity_torch(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    rho: torch.Tensor,
    thickness: torch.Tensor,
    Q_alpha: torch.Tensor,
    Q_beta: torch.Tensor,
    p: float | torch.Tensor,
    omega: torch.Tensor,
    free_surface: bool = False,
    solver: str = "riccati",
) -> torch.Tensor:
    """Compute plane-wave reflectivity using the Global Matrix Method (PyTorch).

    Differentiable via ``torch.autograd``: ``torch.linalg.solve`` provides
    implicit differentiation through the linear system automatically.

    Args:
        alpha: P-wave velocities, shape (n_layers,).
        beta: S-wave velocities (0 for acoustic), shape (n_layers,).
        rho: Densities, shape (n_layers,).
        thickness: Layer thicknesses (large finite value for half-space), shape (n_layers,).
        Q_alpha: P-wave quality factors, shape (n_layers,).
        Q_beta: S-wave quality factors, shape (n_layers,).
        p: Horizontal slowness (ray parameter), scalar.
        omega: Angular frequencies, shape (nfreq,). Must not include DC.
        free_surface: Include free-surface reverberations.
        solver: ``"riccati"`` (default, O(N)) or ``"dense"`` (O(N^3)).

    Returns:
        Complex PP reflectivity, shape (nfreq,).
    """
    if not isinstance(p, torch.Tensor):
        p = torch.tensor(p, dtype=_FDTYPE)
    cp = p.to(_CDTYPE)

    nlayer = alpha.shape[0]

    # Complex slownesses
    s_p = complex_slowness_torch(alpha, Q_alpha)
    beta_pos = beta > 0
    safe_beta = torch.where(beta_pos, beta, torch.ones_like(beta))
    safe_Q_beta = torch.where(beta_pos, Q_beta, torch.ones_like(Q_beta))
    s_s_all = complex_slowness_torch(safe_beta, safe_Q_beta)
    s_s = torch.where(beta_pos, s_s_all, torch.zeros_like(s_s_all))

    s_s_nonzero = s_s.abs() > 0
    safe_s_s = torch.where(s_s_nonzero, s_s, torch.ones_like(s_s))
    beta_c = torch.where(s_s_nonzero, 1.0 / safe_s_s, torch.zeros_like(s_s))

    # Vertical slownesses
    eta = vertical_slowness_torch(s_p, cp)
    neta_all = vertical_slowness_torch(s_s, cp)
    ocean_mask = torch.zeros(nlayer, dtype=torch.bool)
    ocean_mask[0] = True
    neta = torch.where(ocean_mask, torch.zeros_like(neta_all), neta_all)

    if solver == "riccati":
        E_d, E_u, phase_d, e_d_oc, e_u_oc, e0 = _compute_eigenvectors_torch(
            nlayer, eta, neta, rho, beta_c, thickness, p, omega
        )
        R, _U0_P = riccati_sweep_torch(nlayer, E_d, E_u, phase_d, e_d_oc, e_u_oc, e0)
    elif solver == "dense":
        G, b_vec, N = _build_system_torch(
            nlayer, eta, neta, rho, beta_c, thickness, p, omega
        )
        x = torch.linalg.solve(G, b_vec)
        e0 = torch.exp(1j * omega * eta[0] * thickness[0])
        R = e0 * x[:, 0]
    else:
        msg = f"Unknown solver {solver!r}, expected 'riccati' or 'dense'"
        raise ValueError(msg) from None

    if free_surface:
        R = R / (1.0 + R)

    return R


def gmm_jacobian(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    rho: torch.Tensor,
    thickness: torch.Tensor,
    Q_alpha: torch.Tensor,
    Q_beta: torch.Tensor,
    p: float,
    omega: torch.Tensor,
    free_surface: bool = False,
) -> torch.Tensor:
    r"""Compute the Jacobian :math:`J_{ij} = \partial R(\omega_i) / \partial m_j`.

    Uses ``torch.func.jacrev`` (vectorized reverse-mode AD) through the
    GMM linear solve.

    Returns:
        Complex Jacobian, shape (nfreq, n_params).
    """
    params = _pack_params(alpha, beta, rho, thickness)

    def _forward(p_vec: torch.Tensor) -> torch.Tensor:
        a, b, r, h = _unpack_params(p_vec, alpha, beta, rho, thickness)
        return gmm_reflectivity_torch(
            a, b, r, h, Q_alpha, Q_beta, p, omega, free_surface=free_surface
        )

    J_re = torch.func.jacrev(lambda pv: _forward(pv).real)(params)
    J_im = torch.func.jacrev(lambda pv: _forward(pv).imag)(params)
    return J_re + 1j * J_im


def gmm_hessian(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    rho: torch.Tensor,
    thickness: torch.Tensor,
    Q_alpha: torch.Tensor,
    Q_beta: torch.Tensor,
    p: float,
    omega: torch.Tensor,
    R_obs: torch.Tensor | None = None,
    free_surface: bool = False,
) -> torch.Tensor:
    r"""Compute the Hessian :math:`H_{ij} = \partial^2 \chi / \partial m_i \partial m_j`.

    The misfit is :math:`\chi = \|R - R_{\rm obs}\|^2`.

    Uses ``torch.func.hessian`` (= ``jacrev(grad)``) for vectorized
    second-order differentiation.

    Args:
        R_obs: Observed reflectivity, shape (nfreq,). Defaults to zeros.

    Returns:
        Real Hessian, shape (n_params, n_params).
    """
    params = _pack_params(alpha, beta, rho, thickness)

    if R_obs is None:
        R_obs = torch.zeros(omega.shape[0], dtype=_CDTYPE)

    def _misfit(p_vec: torch.Tensor) -> torch.Tensor:
        a, b, r, h = _unpack_params(p_vec, alpha, beta, rho, thickness)
        R = gmm_reflectivity_torch(
            a, b, r, h, Q_alpha, Q_beta, p, omega, free_surface=free_surface
        )
        residual = R - R_obs
        return (residual.real**2 + residual.imag**2).sum()

    return torch.func.hessian(_misfit)(params)
