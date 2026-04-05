"""Differentiable Kennett reflectivity in PyTorch.

Mirrors the NumPy implementation in ``kennett_reflectivity.py`` but replaces all
operations with their differentiable ``torch`` equivalents.  This enables
automatic computation of the Jacobian and Hessian via ``torch.autograd``.

Convention: exp(-iωt) inverse Fourier transform, depth positive downward.
Branch cut: Im(η) > 0 for vertical slowness.

All computation uses ``torch.complex128`` (double precision) to match the
existing NumPy reference and avoid evanescent-phase underflow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .layer_model import LayerModel

__all__ = [
    "complex_slowness_torch",
    "vertical_slowness_torch",
    "solid_solid_interface_torch",
    "ocean_bottom_interface_torch",
    "kennett_reflectivity_torch",
    "forward_model_torch",
    "jacobian",
    "hessian",
]

_CDTYPE = torch.complex128
_FDTYPE = torch.float64


# ---------------------------------------------------------------------------
# 1a. Complex slowness (vectorised over layers)
# ---------------------------------------------------------------------------


def complex_slowness_torch(
    velocity: torch.Tensor,
    Q: torch.Tensor,
) -> torch.Tensor:
    """Compute complex slowness for all layers at once.

    Args:
        velocity: Real phase velocities, shape ``(n,)``.
        Q: Quality factors, shape ``(n,)``.

    Returns:
        Complex slowness tensor, shape ``(n,)``, dtype ``complex128``.
    """
    twoQ = 2.0 * Q
    twoQsq = twoQ * twoQ
    denom = (1.0 + twoQsq) * velocity
    real_part = twoQsq / denom
    imag_part = twoQ / denom
    return torch.complex(real_part, imag_part)


# ---------------------------------------------------------------------------
# 1b. Vertical slowness (vectorised, branch-cut safe)
# ---------------------------------------------------------------------------


def vertical_slowness_torch(
    slowness: torch.Tensor,
    p: torch.Tensor,
) -> torch.Tensor:
    """Vertical slowness with Im(η) > 0 branch cut convention.

    Args:
        slowness: Complex slowness, any shape.
        p: Horizontal slowness (scalar or broadcastable).

    Returns:
        Vertical slowness η with ``Im(η) > 0``, same shape as *slowness*.
    """
    T = (slowness + p) * (slowness - p)
    eta = torch.sqrt(T)
    # Enforce Im(η) > 0 (matches Fortran .LE. 0.0 convention)
    eta = torch.where(eta.imag <= 0, -eta, eta)
    return eta


# ---------------------------------------------------------------------------
# 1c. Solid-solid interface scattering matrices
# ---------------------------------------------------------------------------


def solid_solid_interface_torch(
    p: torch.Tensor,
    eta1: torch.Tensor,
    neta1: torch.Tensor,
    rho1: torch.Tensor,
    beta1: torch.Tensor,
    eta2: torch.Tensor,
    neta2: torch.Tensor,
    rho2: torch.Tensor,
    beta2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Modified P-SV scattering coefficients at a solid-solid interface.

    All inputs are scalar tensors (complex128 or float64).

    Returns:
        ``(Rd, Ru, Tu, Td)`` each shape ``(2, 2)`` complex128.
    """
    rtrho1 = torch.sqrt(rho1.to(_CDTYPE))
    rtrho2 = torch.sqrt(rho2.to(_CDTYPE))
    rteta1 = torch.sqrt(eta1)
    rteta2 = torch.sqrt(eta2)
    rtneta1 = torch.sqrt(neta1)
    rtneta2 = torch.sqrt(neta2)
    rtza1 = rteta1 * rtrho1
    rtza2 = rteta2 * rtrho2
    rtzb1 = rtneta1 * rtrho1
    rtzb2 = rtneta2 * rtrho2

    psq = (p * p).to(_CDTYPE)
    crho1 = rho1.to(_CDTYPE)
    crho2 = rho2.to(_CDTYPE)
    drho = crho2 - crho1
    dmu = crho2 * beta2 * beta2 - crho1 * beta1 * beta1

    d = 2.0 * dmu
    psqd = psq * d
    a = drho - psqd
    b = crho2 - psqd
    c = crho1 + psqd

    E = b * eta1 + c * eta2
    F = b * neta1 + c * neta2
    G = a - d * eta1 * neta2
    H = a - d * eta2 * neta1

    Det = E * F + G * H * psq

    E = E / Det
    F = F / Det
    G = G / Det
    H = H / Det

    Q_val = (b * eta1 - c * eta2) * F
    R_val = (a + d * eta1 * neta2) * H * psq
    S_val = (a * b + c * d * eta2 * neta2) * p.to(_CDTYPE) / Det
    T_val = (b * neta1 - c * neta2) * E
    U_val = (a + d * eta2 * neta1) * G * psq
    V_val = (a * c + b * d * eta1 * neta1) * p.to(_CDTYPE) / Det

    m2ci = torch.tensor(-2.0j, dtype=_CDTYPE)

    Rd = torch.stack(
        [
            torch.stack([Q_val - R_val, m2ci * rteta1 * rtneta1 * S_val]),
            torch.stack([m2ci * rteta1 * rtneta1 * S_val, T_val - U_val]),
        ]
    )

    Td = torch.stack(
        [
            torch.stack(
                [2 * rtza1 * rtza2 * F, m2ci * rtzb1 * rtza2 * G * p.to(_CDTYPE)]
            ),
            torch.stack(
                [m2ci * rtza1 * rtzb2 * H * p.to(_CDTYPE), 2 * rtzb1 * rtzb2 * E]
            ),
        ]
    )

    Tu = Td.T

    Ru = torch.stack(
        [
            torch.stack([-(Q_val + U_val), m2ci * rteta2 * rtneta2 * V_val]),
            torch.stack([m2ci * rteta2 * rtneta2 * V_val, -(T_val + R_val)]),
        ]
    )

    return Rd, Ru, Tu, Td


# ---------------------------------------------------------------------------
# 1d. Ocean-bottom interface scattering matrices
# ---------------------------------------------------------------------------


def ocean_bottom_interface_torch(
    p: torch.Tensor,
    eta1: torch.Tensor,
    rho1: torch.Tensor,
    eta2: torch.Tensor,
    neta2: torch.Tensor,
    rho2: torch.Tensor,
    beta2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scattering coefficients at acoustic-elastic (ocean-bottom) interface.

    Returns:
        ``(Rd, Ru, Tu, Td)`` each shape ``(2, 2)`` complex128.
    """
    rtrho1 = torch.sqrt(rho1.to(_CDTYPE))
    rtrho2 = torch.sqrt(rho2.to(_CDTYPE))
    rteta1 = torch.sqrt(eta1)
    rteta2 = torch.sqrt(eta2)
    rtneta2 = torch.sqrt(neta2)
    rtza1 = rtrho1 * rteta1
    rtza2 = rtrho2 * rteta2
    rtzb2 = rtrho2 * rtneta2

    psq = (p * p).to(_CDTYPE)
    crho1 = rho1.to(_CDTYPE)
    crho2 = rho2.to(_CDTYPE)
    drho = crho2 - crho1
    dmu = crho2 * beta2 * beta2

    d = 2.0 * dmu
    psqd = psq * d
    a = drho - psqd
    b = crho2 - psqd
    c = crho1 + psqd

    E = b * eta1 + c * eta2
    F = b
    G = a - d * eta1 * neta2
    H = -d * eta2

    Det = E * F + G * H * psq

    E = E / Det
    F = F / Det
    G = G / Det
    H = H / Det

    T1 = (b * eta1 - c * eta2) * F
    T2 = (a + d * eta1 * neta2) * H * psq
    T4 = b * E
    T5 = d * eta2 * G * psq
    T6 = b * d * eta1 * p.to(_CDTYPE) / Det

    mci = torch.tensor(-1.0j, dtype=_CDTYPE)
    zero = torch.tensor(0.0 + 0.0j, dtype=_CDTYPE)

    PdPu = T1 - T2
    PdPd = 2.0 * rtza1 * rtza2 * F
    PdSd = 2.0 * mci * rtza1 * rtzb2 * H * p.to(_CDTYPE)
    PuPu = PdPd
    SuPu = PdSd
    PuPd = -(T1 + T5)
    PuSd = 2.0 * mci * rteta2 * rtneta2 * T6
    SuPd = PuSd
    SuSd = -(T2 + T4)

    Rd = torch.stack(
        [
            torch.stack([PdPu, zero]),
            torch.stack([zero, zero]),
        ]
    )

    Td = torch.stack(
        [
            torch.stack([PdPd, zero]),
            torch.stack([PdSd, zero]),
        ]
    )

    Tu = torch.stack(
        [
            torch.stack([PuPu, SuPu]),
            torch.stack([zero, zero]),
        ]
    )

    Ru = torch.stack(
        [
            torch.stack([PuPd, SuPd]),
            torch.stack([PuSd, SuSd]),
        ]
    )

    return Rd, Ru, Tu, Td


# ---------------------------------------------------------------------------
# 1e. Kennett reflectivity (core recursive algorithm)
# ---------------------------------------------------------------------------


def _batch_inv2x2(M: torch.Tensor) -> torch.Tensor:
    """Analytical inverse of batched 2x2 matrices without in-place ops.

    Args:
        M: shape ``(nfreq, 2, 2)`` complex128.

    Returns:
        Inverse, same shape.
    """
    a = M[:, 0, 0]
    b = M[:, 0, 1]
    c = M[:, 1, 0]
    d = M[:, 1, 1]
    det = a * d - b * c
    row0 = torch.stack([d / det, -b / det], dim=-1)
    row1 = torch.stack([-c / det, a / det], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def kennett_reflectivity_torch(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    rho: torch.Tensor,
    thickness: torch.Tensor,
    Q_alpha: torch.Tensor,
    Q_beta: torch.Tensor,
    p: float | torch.Tensor,
    omega: torch.Tensor,
    free_surface: bool = False,
) -> torch.Tensor:
    """Kennett recursive reflectivity, differentiable via ``torch.autograd``.

    Args:
        alpha: P-wave velocities, shape ``(n_layers,)``.
        beta: S-wave velocities (0 for acoustic), shape ``(n_layers,)``.
        rho: Densities, shape ``(n_layers,)``.
        thickness: Layer thicknesses (use large finite value for half-space),
            shape ``(n_layers,)``.
        Q_alpha: P-wave quality factors, shape ``(n_layers,)``.
        Q_beta: S-wave quality factors, shape ``(n_layers,)``.
        p: Horizontal slowness (ray parameter), scalar.
        omega: Angular frequencies, shape ``(nfreq,)``.  Must not include DC.
        free_surface: Include free-surface reverberations.

    Returns:
        Complex PP reflectivity at each frequency, shape ``(nfreq,)``.
    """
    if not isinstance(p, torch.Tensor):
        p = torch.tensor(p, dtype=_FDTYPE)
    cp = p.to(_CDTYPE)

    nlayer = alpha.shape[0]
    nfreq = omega.shape[0]

    # --- Complex slownesses (vectorised) ---
    s_p = complex_slowness_torch(alpha, Q_alpha)

    # S-wave: use torch.where for acoustic layers (beta == 0)
    beta_pos = beta > 0
    # Compute for all layers, then zero out acoustic ones
    safe_beta = torch.where(beta_pos, beta, torch.ones_like(beta))
    safe_Q_beta = torch.where(beta_pos, Q_beta, torch.ones_like(Q_beta))
    s_s_all = complex_slowness_torch(safe_beta, safe_Q_beta)
    s_s = torch.where(beta_pos, s_s_all, torch.zeros_like(s_s_all))

    # Complex S-wave velocity = 1/s_s where s_s != 0
    s_s_nonzero = s_s.abs() > 0
    safe_s_s = torch.where(s_s_nonzero, s_s, torch.ones_like(s_s))
    beta_c = torch.where(s_s_nonzero, 1.0 / safe_s_s, torch.zeros_like(s_s))

    # --- Vertical slownesses ---
    eta = vertical_slowness_torch(s_p, cp)

    # S-wave vertical slowness: 0 for ocean (layer 0), computed for rest
    neta_all = vertical_slowness_torch(s_s, cp)
    ocean_mask = torch.zeros(nlayer, dtype=torch.bool)
    ocean_mask[0] = True
    neta = torch.where(ocean_mask, torch.zeros_like(neta_all), neta_all)

    # --- Phase factors ---
    # Identify finite-thickness layers (half-space has very large thickness)
    finite_mask = thickness < 1e30
    safe_thick = torch.where(finite_mask, thickness, torch.zeros_like(thickness))

    # ea[i, w] = exp(i * omega[w] * eta[i] * h[i])  for finite layers, else 1
    phase_p = 1j * omega.unsqueeze(0) * (eta * safe_thick).unsqueeze(1)
    ea = torch.where(
        finite_mask.unsqueeze(1),
        torch.exp(phase_p),
        torch.ones_like(phase_p),
    )

    phase_s = 1j * omega.unsqueeze(0) * (neta * safe_thick).unsqueeze(1)
    eb = torch.where(
        finite_mask.unsqueeze(1),
        torch.exp(phase_s),
        torch.ones_like(phase_s),
    )

    # --- Scattering coefficients at all interfaces ---
    # Interface 0: ocean-bottom (acoustic-elastic)
    scat_Rd = []
    scat_Ru = []
    scat_Tu = []
    scat_Td = []

    Rd0, Ru0, Tu0, Td0 = ocean_bottom_interface_torch(
        p=p,
        eta1=eta[0],
        rho1=rho[0],
        eta2=eta[1],
        neta2=neta[1],
        rho2=rho[1],
        beta2=beta_c[1],
    )
    scat_Rd.append(Rd0)
    scat_Ru.append(Ru0)
    scat_Tu.append(Tu0)
    scat_Td.append(Td0)

    # Interfaces 1..nlayer-2: solid-solid
    for il in range(1, nlayer - 1):
        Rd_i, Ru_i, Tu_i, Td_i = solid_solid_interface_torch(
            p=p,
            eta1=eta[il],
            neta1=neta[il],
            rho1=rho[il],
            beta1=beta_c[il],
            eta2=eta[il + 1],
            neta2=neta[il + 1],
            rho2=rho[il + 1],
            beta2=beta_c[il + 1],
        )
        scat_Rd.append(Rd_i)
        scat_Ru.append(Ru_i)
        scat_Tu.append(Tu_i)
        scat_Td.append(Td_i)

    # --- Kennett upward sweep ---
    RRd = torch.zeros((nfreq, 2, 2), dtype=_CDTYPE)
    I2 = torch.eye(2, dtype=_CDTYPE)
    n_interfaces = nlayer - 1

    for iface in range(n_interfaces - 1, -1, -1):
        i_below = iface + 1

        Rd_if = scat_Rd[iface]
        Ru_if = scat_Ru[iface]
        Tu_if = scat_Tu[iface]
        Td_if = scat_Td[iface]

        # Phase factors for the layer below
        eaea = ea[i_below, :] ** 2  # (nfreq,)
        ebeb = eb[i_below, :] ** 2
        eaeb = ea[i_below, :] * eb[i_below, :]

        # MT = E * RRd * E  (no in-place ops)
        MT = torch.stack(
            [
                torch.stack([eaea * RRd[:, 0, 0], eaeb * RRd[:, 0, 1]], dim=-1),
                torch.stack([eaeb * RRd[:, 1, 0], ebeb * RRd[:, 1, 1]], dim=-1),
            ],
            dim=-2,
        )

        # U = (I - Ru @ MT)^{-1}
        RuMT = torch.matmul(Ru_if.unsqueeze(0), MT)
        I_minus_RuMT = I2.unsqueeze(0) - RuMT
        U = _batch_inv2x2(I_minus_RuMT)

        # RRd = Rd + Tu @ MT @ U @ Td
        MTU = torch.matmul(MT, U)
        TuMTU = torch.matmul(Tu_if.unsqueeze(0), MTU)
        TuMTUTd = torch.matmul(TuMTU, Td_if.unsqueeze(0))
        RRd = Rd_if.unsqueeze(0) + TuMTUTd

    # --- Extract PP reflectivity ---
    RRd_PP = RRd[:, 0, 0]
    eaea_ocean = ea[0, :] ** 2

    if not free_surface:
        R = eaea_ocean * RRd_PP
    else:
        numerator = eaea_ocean * RRd_PP
        denominator = 1.0 + eaea_ocean * RRd_PP
        R = numerator / denominator

    return R


# ---------------------------------------------------------------------------
# 1f. Convenience wrapper: forward model → seismogram
# ---------------------------------------------------------------------------


def forward_model_torch(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    rho: torch.Tensor,
    thickness: torch.Tensor,
    Q_alpha: torch.Tensor,
    Q_beta: torch.Tensor,
    p: float,
    T: float = 64.0,
    nw: int = 2048,
    free_surface: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a differentiable synthetic seismogram.

    Chains: parameters → reflectivity → source convolution → IFFT.

    Returns:
        ``(time, seismogram)`` both real-valued tensors of shape ``(2*nw,)``.
    """
    dw = 2.0 * torch.pi / T
    nwm = nw - 1
    wmax = nw * dw

    omega = torch.arange(1, nwm + 1, dtype=_FDTYPE) * dw

    # Ricker source spectrum
    alpha_src = 0.1 * wmax
    T0 = 5.0 / (alpha_src * (2.0**0.5))
    Z = (omega**2) / (4.0 * alpha_src**2)
    S = Z * torch.exp(-Z + 1j * omega * T0)

    R = kennett_reflectivity_torch(
        alpha,
        beta,
        rho,
        thickness,
        Q_alpha,
        Q_beta,
        p,
        omega,
        free_surface=free_surface,
    )

    Y = R * S

    nt = 2 * nw
    Uwk = torch.zeros(nt, dtype=_CDTYPE)
    Uwk[1:nw] = Y
    Uwk[nw + 1 :] = torch.conj(torch.flip(Y, [0]))

    seismogram_c = torch.fft.fft(Uwk)

    dt = T / float(nt)
    time = torch.arange(nt, dtype=_FDTYPE) * dt
    seismogram = seismogram_c.real * torch.exp(torch.tensor(0.0, dtype=_FDTYPE) * time)

    return time, seismogram


# ---------------------------------------------------------------------------
# 2. Jacobian and Hessian API
# ---------------------------------------------------------------------------


def _pack_params(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    rho: torch.Tensor,
    thickness: torch.Tensor,
) -> torch.Tensor:
    """Concatenate differentiable sub-ocean parameters into a single vector.

    The parameter vector is::

        m = [Vp_1, ..., Vp_N, Vs_1, ..., Vs_N, ρ_1, ..., ρ_N, h_1, ..., h_{N-1}]

    where layers ``1..N`` are sub-ocean (layer 0 is the ocean, excluded)
    and the half-space thickness (last element) is excluded.
    """
    alpha_sub = alpha[1:]
    beta_sub = beta[1:]
    rho_sub = rho[1:]
    thick_finite = thickness[1:-1]  # exclude ocean (0) and half-space (last)
    return torch.cat([alpha_sub, beta_sub, rho_sub, thick_finite])


def _unpack_params(
    params: torch.Tensor,
    alpha_full: torch.Tensor,
    beta_full: torch.Tensor,
    rho_full: torch.Tensor,
    thickness_full: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack parameter vector back into full model arrays.

    Fixed ocean (layer 0) and half-space thickness are taken from the
    ``*_full`` reference tensors.
    """
    n_sub = alpha_full.shape[0] - 1  # number of sub-ocean layers
    n_thick = n_sub - 1  # finite thickness layers (exclude half-space)

    idx = 0
    alpha_sub = params[idx : idx + n_sub]
    idx += n_sub
    beta_sub = params[idx : idx + n_sub]
    idx += n_sub
    rho_sub = params[idx : idx + n_sub]
    idx += n_sub
    thick_sub = params[idx : idx + n_thick]

    alpha_out = torch.cat([alpha_full[:1].detach(), alpha_sub])
    beta_out = torch.cat([beta_full[:1].detach(), beta_sub])
    rho_out = torch.cat([rho_full[:1].detach(), rho_sub])
    thickness_out = torch.cat(
        [
            thickness_full[:1].detach(),
            thick_sub,
            thickness_full[-1:].detach(),
        ]
    )
    return alpha_out, beta_out, rho_out, thickness_out


def jacobian(
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

    Uses ``torch.func.jacrev`` (vectorized reverse-mode AD) for exact
    differentiation through the Kennett recursion.

    Returns:
        Complex Jacobian tensor of shape ``(nfreq, n_params)`` where
        ``n_params = 4*N - 1`` for *N* sub-ocean layers.
    """
    params = _pack_params(alpha, beta, rho, thickness)

    def _forward(p_vec: torch.Tensor) -> torch.Tensor:
        a, b, r, h = _unpack_params(p_vec, alpha, beta, rho, thickness)
        return kennett_reflectivity_torch(
            a, b, r, h, Q_alpha, Q_beta, p, omega, free_surface=free_surface
        )

    J_re = torch.func.jacrev(lambda pv: _forward(pv).real)(params)
    J_im = torch.func.jacrev(lambda pv: _forward(pv).imag)(params)
    return J_re + 1j * J_im


def hessian(
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

    The misfit is :math:`\chi = \|R - R_{\rm obs}\|^2` (sum of squared
    magnitudes of the complex residual).

    Uses ``torch.func.hessian`` (= ``jacrev(grad)``) for vectorized
    second-order differentiation.

    Args:
        R_obs: Observed reflectivity, shape ``(nfreq,)``.  If *None*,
            uses zeros (useful for testing).

    Returns:
        Real Hessian tensor of shape ``(n_params, n_params)``.
    """
    params = _pack_params(alpha, beta, rho, thickness)

    if R_obs is None:
        R_obs = torch.zeros(omega.shape[0], dtype=_CDTYPE)

    def _misfit(p_vec: torch.Tensor) -> torch.Tensor:
        a, b, r, h = _unpack_params(p_vec, alpha, beta, rho, thickness)
        R = kennett_reflectivity_torch(
            a, b, r, h, Q_alpha, Q_beta, p, omega, free_surface=free_surface
        )
        residual = R - R_obs
        return (residual.real**2 + residual.imag**2).sum()

    return torch.func.hessian(_misfit)(params)


# ---------------------------------------------------------------------------
# Utility: convert LayerModel to torch tensors
# ---------------------------------------------------------------------------


def model_to_tensors(
    model: LayerModel,
    requires_grad: bool = False,
) -> dict[str, torch.Tensor]:
    """Convert a NumPy ``LayerModel`` to a dict of torch tensors.

    Args:
        model: The NumPy layer model.
        requires_grad: Whether the sub-ocean parameters require gradients.

    Returns:
        Dict with keys ``alpha``, ``beta``, ``rho``, ``thickness``,
        ``Q_alpha``, ``Q_beta``.  The half-space thickness is replaced
        with a large finite sentinel (``1e30``).
    """
    import numpy as np

    thick = model.thickness.copy()
    thick[np.isinf(thick)] = 1e30

    tensors = {}
    for name, arr in [
        ("alpha", model.alpha),
        ("beta", model.beta),
        ("rho", model.rho),
        ("thickness", thick),
        ("Q_alpha", model.Q_alpha),
        ("Q_beta", model.Q_beta),
    ]:
        t = torch.tensor(arr, dtype=_FDTYPE)
        if requires_grad and name in ("alpha", "beta", "rho", "thickness"):
            t = t.requires_grad_(True)
        tensors[name] = t
    return tensors
