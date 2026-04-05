"""Analytical Fréchet derivatives of Kennett reflectivity.

Independent re-derivation of Dietrich & Kormendi (1990) via tangent-linear
(forward-mode) differentiation of the discrete Kennett recursion.
Uses only NumPy — no automatic differentiation.

The tangent-linear approach: for each model parameter m_j, propagate
dm_j = 1 through the entire computation chain:

    alpha_j → s_p_j → eta_j → (scattering matrices, phase factors) → Kennett recursion → R

Each line of the forward computation gets a corresponding "tangent" line
computing the derivative via the chain rule.

Mathematical identity
---------------------
For infinitesimal perturbations, this tangent-linear derivative equals:
  - The Dietrich & Kormendi (1990) analytical Fréchet derivative
  - The torch.autograd Jacobian
  - The finite-difference Jacobian (to O(δ²))

All four should agree. The first two are exact; the third is our test.
"""

import numpy as np

from .layer_model import complex_slowness, vertical_slowness

__all__ = ["frechet_kennett"]


# ---------------------------------------------------------------------------
# Tangent-linear of solid-solid interface scattering matrices
# ---------------------------------------------------------------------------


def _tl_solid_solid(
    p: float,
    eta1: complex,
    neta1: complex,
    rho1: float,
    beta1: complex,
    eta2: complex,
    neta2: complex,
    rho2: float,
    beta2: complex,
    # Tangent inputs
    deta1: complex,
    dneta1: complex,
    drho1: float,
    dbeta1: complex,
    deta2: complex,
    dneta2: complex,
    drho2: float,
    dbeta2: complex,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Tangent-linear of solid_solid_interface.

    Returns (Rd, Ru, Tu, Td, dRd, dRu, dTu, dTd).
    Forward values + their tangents, each shape (2, 2).
    """
    # --- Forward + tangent, line by line ---

    # Root quantities
    rtrho1 = np.sqrt(complex(rho1))
    drtrho1 = complex(drho1) / (2.0 * rtrho1)

    rtrho2 = np.sqrt(complex(rho2))
    drtrho2 = complex(drho2) / (2.0 * rtrho2)

    rteta1 = np.sqrt(eta1)
    drteta1 = deta1 / (2.0 * rteta1)

    rteta2 = np.sqrt(eta2)
    drteta2 = deta2 / (2.0 * rteta2)

    rtneta1 = np.sqrt(neta1)
    drtneta1 = dneta1 / (2.0 * rtneta1)

    rtneta2 = np.sqrt(neta2)
    drtneta2 = dneta2 / (2.0 * rtneta2)

    # Modified impedances
    rtza1 = rteta1 * rtrho1
    drtza1 = drteta1 * rtrho1 + rteta1 * drtrho1

    rtza2 = rteta2 * rtrho2
    drtza2 = drteta2 * rtrho2 + rteta2 * drtrho2

    rtzb1 = rtneta1 * rtrho1
    drtzb1 = drtneta1 * rtrho1 + rtneta1 * drtrho1

    rtzb2 = rtneta2 * rtrho2
    drtzb2 = drtneta2 * rtrho2 + rtneta2 * drtrho2

    # Scalar quantities (dp = 0 since p is not a model parameter)
    psq = complex(p * p)
    crho1 = complex(rho1)
    dcrho1 = complex(drho1)
    crho2 = complex(rho2)
    dcrho2 = complex(drho2)

    drho_v = crho2 - crho1
    ddrho_v = dcrho2 - dcrho1

    dmu = crho2 * beta2 * beta2 - crho1 * beta1 * beta1
    ddmu = (
        dcrho2 * beta2 * beta2
        + crho2 * 2.0 * beta2 * dbeta2
        - dcrho1 * beta1 * beta1
        - crho1 * 2.0 * beta1 * dbeta1
    )

    d_v = 2.0 * dmu
    dd_v = 2.0 * ddmu

    psqd = psq * d_v
    dpsqd = psq * dd_v

    a = drho_v - psqd
    da = ddrho_v - dpsqd

    b = crho2 - psqd
    db = dcrho2 - dpsqd

    c = crho1 + psqd
    dc = dcrho1 + dpsqd

    # E, F, G, H (before Det division)
    Ev = b * eta1 + c * eta2
    dEv = db * eta1 + b * deta1 + dc * eta2 + c * deta2

    Fv = b * neta1 + c * neta2
    dFv = db * neta1 + b * dneta1 + dc * neta2 + c * dneta2

    Gv = a - d_v * eta1 * neta2
    dGv = da - dd_v * eta1 * neta2 - d_v * deta1 * neta2 - d_v * eta1 * dneta2

    Hv = a - d_v * eta2 * neta1
    dHv = da - dd_v * eta2 * neta1 - d_v * deta2 * neta1 - d_v * eta2 * dneta1

    Det = Ev * Fv + Gv * Hv * psq
    dDet = dEv * Fv + Ev * dFv + (dGv * Hv + Gv * dHv) * psq

    # Save originals before division
    Det_o = Det

    # Divide by Det: x_new = x_old / Det
    # d(x/Det) = (dx * Det - x * dDet) / Det^2
    Det2 = Det_o * Det_o

    Ev_o, dEv_o = Ev, dEv
    Ev = Ev_o / Det_o
    dEv = (dEv_o * Det_o - Ev_o * dDet) / Det2

    Fv_o, dFv_o = Fv, dFv
    Fv = Fv_o / Det_o
    dFv = (dFv_o * Det_o - Fv_o * dDet) / Det2

    Gv_o, dGv_o = Gv, dGv
    Gv = Gv_o / Det_o
    dGv = (dGv_o * Det_o - Gv_o * dDet) / Det2

    Hv_o, dHv_o = Hv, dHv
    Hv = Hv_o / Det_o
    dHv = (dHv_o * Det_o - Hv_o * dDet) / Det2

    # Intermediate scattering quantities
    bec = b * eta1 - c * eta2
    dbec = db * eta1 + b * deta1 - dc * eta2 - c * deta2
    Q_v = bec * Fv
    dQ_v = dbec * Fv + bec * dFv

    aden = a + d_v * eta1 * neta2
    daden = da + dd_v * eta1 * neta2 + d_v * deta1 * neta2 + d_v * eta1 * dneta2
    R_v = aden * Hv * psq
    dR_v = (daden * Hv + aden * dHv) * psq

    # S_val uses original Det (before division)
    ab_cd = a * b + c * d_v * eta2 * neta2
    dab_cd = (
        da * b
        + a * db
        + dc * d_v * eta2 * neta2
        + c * dd_v * eta2 * neta2
        + c * d_v * deta2 * neta2
        + c * d_v * eta2 * dneta2
    )
    S_v = ab_cd * p / Det_o
    dS_v = (dab_cd * Det_o - ab_cd * dDet) * p / Det2

    bnc = b * neta1 - c * neta2
    dbnc = db * neta1 + b * dneta1 - dc * neta2 - c * dneta2
    T_v = bnc * Ev
    dT_v = dbnc * Ev + bnc * dEv

    aden2 = a + d_v * eta2 * neta1
    daden2 = da + dd_v * eta2 * neta1 + d_v * deta2 * neta1 + d_v * eta2 * dneta1
    U_v = aden2 * Gv * psq
    dU_v = (daden2 * Gv + aden2 * dGv) * psq

    ac_bd = a * c + b * d_v * eta1 * neta1
    dac_bd = (
        da * c
        + a * dc
        + db * d_v * eta1 * neta1
        + b * dd_v * eta1 * neta1
        + b * d_v * deta1 * neta1
        + b * d_v * eta1 * dneta1
    )
    V_v = ac_bd * p / Det_o
    dV_v = (dac_bd * Det_o - ac_bd * dDet) * p / Det2

    m2ci = complex(0.0, -2.0)

    # --- Build forward matrices ---
    Rd = np.array(
        [
            [Q_v - R_v, m2ci * rteta1 * rtneta1 * S_v],
            [m2ci * rteta1 * rtneta1 * S_v, T_v - U_v],
        ],
        dtype=np.complex128,
    )

    Td = np.array(
        [
            [2 * rtza1 * rtza2 * Fv, m2ci * rtzb1 * rtza2 * Gv * p],
            [m2ci * rtza1 * rtzb2 * Hv * p, 2 * rtzb1 * rtzb2 * Ev],
        ],
        dtype=np.complex128,
    )

    Tu = Td.T.copy()

    Ru = np.array(
        [
            [-(Q_v + U_v), m2ci * rteta2 * rtneta2 * V_v],
            [m2ci * rteta2 * rtneta2 * V_v, -(T_v + R_v)],
        ],
        dtype=np.complex128,
    )

    # --- Build tangent matrices ---
    # dRd
    dRd01 = m2ci * (
        drteta1 * rtneta1 * S_v + rteta1 * drtneta1 * S_v + rteta1 * rtneta1 * dS_v
    )
    dRd = np.array(
        [
            [dQ_v - dR_v, dRd01],
            [dRd01, dT_v - dU_v],
        ],
        dtype=np.complex128,
    )

    # dTd
    dTd00 = 2 * (drtza1 * rtza2 * Fv + rtza1 * drtza2 * Fv + rtza1 * rtza2 * dFv)
    dTd01 = m2ci * (drtzb1 * rtza2 * Gv + rtzb1 * drtza2 * Gv + rtzb1 * rtza2 * dGv) * p
    dTd10 = m2ci * (drtza1 * rtzb2 * Hv + rtza1 * drtzb2 * Hv + rtza1 * rtzb2 * dHv) * p
    dTd11 = 2 * (drtzb1 * rtzb2 * Ev + rtzb1 * drtzb2 * Ev + rtzb1 * rtzb2 * dEv)
    dTd = np.array([[dTd00, dTd01], [dTd10, dTd11]], dtype=np.complex128)

    dTu = dTd.T.copy()

    # dRu
    dRu01 = m2ci * (
        drteta2 * rtneta2 * V_v + rteta2 * drtneta2 * V_v + rteta2 * rtneta2 * dV_v
    )
    dRu = np.array(
        [
            [-(dQ_v + dU_v), dRu01],
            [dRu01, -(dT_v + dR_v)],
        ],
        dtype=np.complex128,
    )

    return Rd, Ru, Tu, Td, dRd, dRu, dTu, dTd


# ---------------------------------------------------------------------------
# Tangent-linear of ocean-bottom interface scattering matrices
# ---------------------------------------------------------------------------


def _tl_ocean_bottom(
    p: float,
    eta1: complex,
    rho1: float,
    eta2: complex,
    neta2: complex,
    rho2: float,
    beta2: complex,
    # Tangent inputs (ocean layer is fixed, so deta1=drho1=0)
    deta2: complex,
    dneta2: complex,
    drho2: float,
    dbeta2: complex,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Tangent-linear of ocean_bottom_interface.

    Ocean (layer 1) parameters are fixed: deta1 = drho1 = 0.
    Returns (Rd, Ru, Tu, Td, dRd, dRu, dTu, dTd).
    """
    rtrho1 = np.sqrt(complex(rho1))
    drtrho1 = 0.0 + 0.0j  # ocean fixed

    rtrho2 = np.sqrt(complex(rho2))
    drtrho2 = complex(drho2) / (2.0 * rtrho2)

    rteta1 = np.sqrt(eta1)
    drteta1 = 0.0 + 0.0j  # ocean fixed

    rteta2 = np.sqrt(eta2)
    drteta2 = deta2 / (2.0 * rteta2)

    rtneta2 = np.sqrt(neta2)
    drtneta2 = dneta2 / (2.0 * rtneta2)

    rtza1 = rtrho1 * rteta1
    drtza1 = 0.0 + 0.0j  # ocean fixed

    rtza2 = rtrho2 * rteta2
    drtza2 = drtrho2 * rteta2 + rtrho2 * drteta2

    rtzb2 = rtrho2 * rtneta2
    drtzb2 = drtrho2 * rtneta2 + rtrho2 * drtneta2

    psq = complex(p * p)
    crho1 = complex(rho1)
    crho2 = complex(rho2)
    dcrho2 = complex(drho2)

    drho_v = crho2 - crho1
    ddrho_v = dcrho2  # dcrho1 = 0

    dmu = crho2 * beta2 * beta2
    ddmu = dcrho2 * beta2 * beta2 + crho2 * 2.0 * beta2 * dbeta2

    d_v = 2.0 * dmu
    dd_v = 2.0 * ddmu

    psqd = psq * d_v
    dpsqd = psq * dd_v

    a = drho_v - psqd
    da = ddrho_v - dpsqd

    b = crho2 - psqd
    db = dcrho2 - dpsqd

    c = crho1 + psqd
    dc = dpsqd  # dcrho1 = 0

    # Ocean-bottom special forms
    Ev = b * eta1 + c * eta2
    dEv = db * eta1 + dc * eta2 + c * deta2  # deta1 = 0

    Fv = b  # neta1 = 0 for ocean
    dFv = db

    Gv = a - d_v * eta1 * neta2
    dGv = da - dd_v * eta1 * neta2 - d_v * eta1 * dneta2  # deta1 = 0

    Hv = -d_v * eta2  # neta1 = 0
    dHv = -dd_v * eta2 - d_v * deta2

    Det = Ev * Fv + Gv * Hv * psq
    dDet = dEv * Fv + Ev * dFv + (dGv * Hv + Gv * dHv) * psq

    Det_o = Det
    Det2 = Det_o * Det_o

    Ev_o, dEv_o = Ev, dEv
    Ev = Ev_o / Det_o
    dEv = (dEv_o * Det_o - Ev_o * dDet) / Det2

    Fv_o, dFv_o = Fv, dFv
    Fv = Fv_o / Det_o
    dFv = (dFv_o * Det_o - Fv_o * dDet) / Det2

    Gv_o, dGv_o = Gv, dGv
    Gv = Gv_o / Det_o
    dGv = (dGv_o * Det_o - Gv_o * dDet) / Det2

    Hv_o, dHv_o = Hv, dHv
    Hv = Hv_o / Det_o
    dHv = (dHv_o * Det_o - Hv_o * dDet) / Det2

    T1 = (b * eta1 - c * eta2) * Fv
    dT1_inner = db * eta1 - dc * eta2 - c * deta2  # deta1 = 0
    dT1 = dT1_inner * Fv + (b * eta1 - c * eta2) * dFv

    T2 = (a + d_v * eta1 * neta2) * Hv * psq
    dT2_inner = da + dd_v * eta1 * neta2 + d_v * eta1 * dneta2
    dT2 = (dT2_inner * Hv + (a + d_v * eta1 * neta2) * dHv) * psq

    T4 = b * Ev
    dT4 = db * Ev + b * dEv

    T5 = d_v * eta2 * Gv * psq
    dT5 = (dd_v * eta2 * Gv + d_v * deta2 * Gv + d_v * eta2 * dGv) * psq

    T6 = b * d_v * eta1 * p / Det_o
    dT6_num = db * d_v * eta1 + b * dd_v * eta1  # deta1 = 0
    dT6 = (dT6_num * Det_o - b * d_v * eta1 * dDet) * p / Det2

    mci = complex(0.0, -1.0)

    PdPu = T1 - T2
    dPdPu = dT1 - dT2

    PdPd = 2.0 * rtza1 * rtza2 * Fv
    dPdPd = 2.0 * (drtza1 * rtza2 * Fv + rtza1 * drtza2 * Fv + rtza1 * rtza2 * dFv)

    PdSd = 2.0 * mci * rtza1 * rtzb2 * Hv * p
    dPdSd = (
        2.0
        * mci
        * (drtza1 * rtzb2 * Hv + rtza1 * drtzb2 * Hv + rtza1 * rtzb2 * dHv)
        * p
    )

    PuPd = -(T1 + T5)
    dPuPd = -(dT1 + dT5)

    PuSd = 2.0 * mci * rteta2 * rtneta2 * T6
    dPuSd = (
        2.0
        * mci
        * (drteta2 * rtneta2 * T6 + rteta2 * drtneta2 * T6 + rteta2 * rtneta2 * dT6)
    )

    SuSd = -(T2 + T4)
    dSuSd = -(dT2 + dT4)

    z = 0.0 + 0.0j
    dz = 0.0 + 0.0j

    Rd = np.array([[PdPu, z], [z, z]], dtype=np.complex128)
    Td = np.array([[PdPd, z], [PdSd, z]], dtype=np.complex128)
    Tu = np.array([[PdPd, PdSd], [z, z]], dtype=np.complex128)
    Ru = np.array([[PuPd, PuSd], [PuSd, SuSd]], dtype=np.complex128)

    dRd = np.array([[dPdPu, dz], [dz, dz]], dtype=np.complex128)
    dTd = np.array([[dPdPd, dz], [dPdSd, dz]], dtype=np.complex128)
    dTu = np.array([[dPdPd, dPdSd], [dz, dz]], dtype=np.complex128)
    dRu = np.array([[dPuPd, dPuSd], [dPuSd, dSuSd]], dtype=np.complex128)

    return Rd, Ru, Tu, Td, dRd, dRu, dTu, dTd


# ---------------------------------------------------------------------------
# Tangent-linear Kennett recursion step
# ---------------------------------------------------------------------------


def _tl_kennett_step(
    Rd: np.ndarray,
    Ru: np.ndarray,
    Tu: np.ndarray,
    Td: np.ndarray,
    dRd: np.ndarray,
    dRu: np.ndarray,
    dTu: np.ndarray,
    dTd: np.ndarray,
    ea: np.ndarray,
    eb: np.ndarray,
    dea: np.ndarray,
    deb: np.ndarray,
    RRd: np.ndarray,
    dRRd: np.ndarray,
) -> np.ndarray:
    """One step of the tangent-linear Kennett recursion.

    Args:
        Rd, Ru, Tu, Td: Scattering matrices at this interface, shape (2, 2).
        dRd, dRu, dTu, dTd: Their tangents, shape (2, 2).
        ea, eb: Phase factors for layer below, shape (nfreq,).
        dea, deb: Phase factor tangents, shape (nfreq,).
        RRd: Cumulative reflection from below, shape (nfreq, 2, 2).
        dRRd: Its tangent, shape (nfreq, 2, 2).

    Returns:
        New dRRd, shape (nfreq, 2, 2).
    """
    nfreq = RRd.shape[0]

    eaea = ea**2
    ebeb = eb**2
    eaeb = ea * eb

    deaea = 2.0 * ea * dea
    debeb = 2.0 * eb * deb
    deaeb = dea * eb + ea * deb

    # MT = E · RRd · E  (forward)
    MT = np.empty((nfreq, 2, 2), dtype=np.complex128)
    MT[:, 0, 0] = eaea * RRd[:, 0, 0]
    MT[:, 0, 1] = eaeb * RRd[:, 0, 1]
    MT[:, 1, 0] = eaeb * RRd[:, 1, 0]
    MT[:, 1, 1] = ebeb * RRd[:, 1, 1]

    # dMT (tangent of phase-shifted reflection)
    dMT = np.empty((nfreq, 2, 2), dtype=np.complex128)
    dMT[:, 0, 0] = deaea * RRd[:, 0, 0] + eaea * dRRd[:, 0, 0]
    dMT[:, 0, 1] = deaeb * RRd[:, 0, 1] + eaeb * dRRd[:, 0, 1]
    dMT[:, 1, 0] = deaeb * RRd[:, 1, 0] + eaeb * dRRd[:, 1, 0]
    dMT[:, 1, 1] = debeb * RRd[:, 1, 1] + ebeb * dRRd[:, 1, 1]

    # U = (I - Ru · MT)^{-1}  (forward)
    I2 = np.eye(2, dtype=np.complex128)
    RuMT = np.einsum("ij,wjk->wik", Ru, MT)
    W = I2[np.newaxis, :, :] - RuMT

    # Batch 2x2 inverse
    a_w = W[:, 0, 0]
    b_w = W[:, 0, 1]
    c_w = W[:, 1, 0]
    d_w = W[:, 1, 1]
    det_w = a_w * d_w - b_w * c_w
    U = np.empty_like(W)
    U[:, 0, 0] = d_w / det_w
    U[:, 0, 1] = -b_w / det_w
    U[:, 1, 0] = -c_w / det_w
    U[:, 1, 1] = a_w / det_w

    # dU = U · (dRu·MT + Ru·dMT) · U
    dRuMT = np.einsum("ij,wjk->wik", dRu, MT)
    RudMT = np.einsum("ij,wjk->wik", Ru, dMT)
    dW_inner = dRuMT + RudMT  # = -(dW), but we need dW^{-1} = U·(dRu·MT+Ru·dMT)·U
    dU = np.einsum("wij,wjk->wik", U, np.einsum("wij,wjk->wik", dW_inner, U))

    # Z = MT · U  (forward)
    Z = np.einsum("wij,wjk->wik", MT, U)

    # dZ = dMT · U + MT · dU
    dZ = np.einsum("wij,wjk->wik", dMT, U) + np.einsum("wij,wjk->wik", MT, dU)

    # RRd_new = Rd + Tu · Z · Td  (forward)
    TuZ = np.einsum("ij,wjk->wik", Tu, Z)
    TuZTd = np.einsum("wij,jk->wik", TuZ, Td)

    # dRRd_new = dRd + dTu·Z·Td + Tu·dZ·Td + Tu·Z·dTd
    term1 = dRd[np.newaxis, :, :]

    dTuZ = np.einsum("ij,wjk->wik", dTu, Z)
    term2 = np.einsum("wij,jk->wik", dTuZ, Td)

    TudZ = np.einsum("ij,wjk->wik", Tu, dZ)
    term3 = np.einsum("wij,jk->wik", TudZ, Td)

    TuZdTd = np.einsum("wij,jk->wik", TuZ, dTd)
    term4 = TuZdTd

    dRRd_new = term1 + term2 + term3 + term4

    return dRRd_new


# ---------------------------------------------------------------------------
# Main function: analytical Fréchet derivative
# ---------------------------------------------------------------------------


def frechet_kennett(
    alpha: np.ndarray,
    beta: np.ndarray,
    rho: np.ndarray,
    thickness: np.ndarray,
    Q_alpha: np.ndarray,
    Q_beta: np.ndarray,
    p: float,
    omega: np.ndarray,
) -> np.ndarray:
    """Analytical Fréchet derivatives of Kennett PP reflectivity.

    Computes J[i,j] = dR(omega_i)/dm_j by tangent-linear (forward-mode)
    differentiation of the discrete Kennett recursion.

    This is an independent derivation equivalent to Dietrich & Kormendi
    (1990), using only NumPy and explicit chain-rule differentiation.

    Args:
        alpha, beta, rho: Shape (n_layers,).
        thickness: Shape (n_layers,). Use np.inf for half-space.
        Q_alpha, Q_beta: Shape (n_layers,).
        p: Ray parameter (scalar).
        omega: Angular frequencies, shape (nfreq,).

    Returns:
        J: Complex Jacobian, shape (nfreq, n_params).
            Parameter order: [Vp_1..N, Vs_1..N, rho_1..N, h_1..N-1]
            where 1..N are sub-ocean layers (0 = ocean, excluded).
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    thickness = np.asarray(thickness, dtype=np.float64)
    Q_alpha = np.asarray(Q_alpha, dtype=np.float64)
    Q_beta = np.asarray(Q_beta, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)

    nlayer = len(alpha)
    nfreq = len(omega)
    n_sub = nlayer - 1  # sub-ocean layers
    n_thick = n_sub - 1  # finite-thickness sub-ocean layers
    n_params = 3 * n_sub + n_thick  # Vp + Vs + rho + h

    cp = complex(p)

    # ================================================================
    # FORWARD PASS — compute and save all intermediate values
    # ================================================================

    # Complex slownesses
    s_p = np.array(
        [complex_slowness(alpha[i], Q_alpha[i]) for i in range(nlayer)],
        dtype=np.complex128,
    )
    s_s = np.zeros(nlayer, dtype=np.complex128)
    for i in range(1, nlayer):
        if beta[i] > 0:
            s_s[i] = complex_slowness(beta[i], Q_beta[i])

    beta_c = np.zeros(nlayer, dtype=np.complex128)
    for i in range(1, nlayer):
        if abs(s_s[i]) > 0:
            beta_c[i] = 1.0 / s_s[i]

    # Vertical slownesses
    eta = np.array(
        [vertical_slowness(s_p[i], cp) for i in range(nlayer)], dtype=np.complex128
    )
    neta = np.zeros(nlayer, dtype=np.complex128)
    for i in range(1, nlayer):
        neta[i] = vertical_slowness(s_s[i], cp)

    # Phase factors
    ea = np.ones((nlayer, nfreq), dtype=np.complex128)
    eb = np.ones((nlayer, nfreq), dtype=np.complex128)
    for i in range(nlayer):
        if not np.isinf(thickness[i]):
            ea[i, :] = np.exp(1j * omega * eta[i] * thickness[i])
            eb[i, :] = np.exp(1j * omega * neta[i] * thickness[i])

    # Scattering coefficients at each interface (forward only)
    from .scattering_matrices import ocean_bottom_interface, solid_solid_interface

    scat_Rd = []
    scat_Ru = []
    scat_Tu = []
    scat_Td = []

    sc = ocean_bottom_interface(p, eta[0], rho[0], eta[1], neta[1], rho[1], beta_c[1])
    scat_Rd.append(sc.Rd)
    scat_Ru.append(sc.Ru)
    scat_Tu.append(sc.Tu)
    scat_Td.append(sc.Td)

    for il in range(1, nlayer - 1):
        sc = solid_solid_interface(
            p,
            eta[il],
            neta[il],
            rho[il],
            beta_c[il],
            eta[il + 1],
            neta[il + 1],
            rho[il + 1],
            beta_c[il + 1],
        )
        scat_Rd.append(sc.Rd)
        scat_Ru.append(sc.Ru)
        scat_Tu.append(sc.Tu)
        scat_Td.append(sc.Td)

    # Forward Kennett recursion — save RRd at each level
    n_ifaces = nlayer - 1
    _zero = np.zeros((nfreq, 2, 2), dtype=np.complex128)
    RRd_save: list[np.ndarray] = [_zero.copy() for _ in range(n_ifaces + 1)]
    RRd = _zero.copy()
    RRd_save[n_ifaces] = RRd.copy()  # below bottom interface

    for iface in range(n_ifaces - 1, -1, -1):
        ib = iface + 1
        eaea = ea[ib, :] ** 2
        ebeb = eb[ib, :] ** 2
        eaeb = ea[ib, :] * eb[ib, :]

        MT = np.empty((nfreq, 2, 2), dtype=np.complex128)
        MT[:, 0, 0] = eaea * RRd[:, 0, 0]
        MT[:, 0, 1] = eaeb * RRd[:, 0, 1]
        MT[:, 1, 0] = eaeb * RRd[:, 1, 0]
        MT[:, 1, 1] = ebeb * RRd[:, 1, 1]

        I2 = np.eye(2, dtype=np.complex128)
        RuMT = np.einsum("ij,wjk->wik", scat_Ru[iface], MT)
        W = I2[np.newaxis, :, :] - RuMT
        aw, bw, cw, dw = W[:, 0, 0], W[:, 0, 1], W[:, 1, 0], W[:, 1, 1]
        det_w = aw * dw - bw * cw
        U = np.empty_like(W)
        U[:, 0, 0] = dw / det_w
        U[:, 0, 1] = -bw / det_w
        U[:, 1, 0] = -cw / det_w
        U[:, 1, 1] = aw / det_w

        MTU = np.einsum("wij,wjk->wik", MT, U)
        TuMTU = np.einsum("ij,wjk->wik", scat_Tu[iface], MTU)
        TuMTUTd = np.einsum("wij,jk->wik", TuMTU, scat_Td[iface])
        RRd = scat_Rd[iface][np.newaxis, :, :] + TuMTUTd
        RRd_save[iface] = RRd.copy()

    # ================================================================
    # TANGENT-LINEAR PASS — one per parameter
    # ================================================================

    J = np.zeros((nfreq, n_params), dtype=np.complex128)

    # Primitive derivative chains:
    # ds_p/dalpha = -s_p / alpha
    # deta/dalpha = (s_p / eta) * (-s_p / alpha) = -s_p^2 / (eta * alpha)
    # ds_s/dbeta = -s_s / beta
    # dneta/dbeta = -s_s^2 / (neta * beta)
    # dbeta_c/dbeta = beta_c / beta
    # dea/deta = i*omega*h * ea   (finite layers)
    # deb/dneta = i*omega*h * eb  (finite layers)
    # dea/dh = i*omega*eta * ea
    # deb/dh = i*omega*neta * eb

    for j_param in range(n_params):
        # Decode parameter index
        if j_param < n_sub:
            # Vp perturbation of sub-ocean layer
            layer = j_param + 1  # sub-ocean layer index (1-based in full model)
            param_type = "alpha"
        elif j_param < 2 * n_sub:
            layer = (j_param - n_sub) + 1
            param_type = "beta"
        elif j_param < 3 * n_sub:
            layer = (j_param - 2 * n_sub) + 1
            param_type = "rho"
        else:
            layer = (j_param - 3 * n_sub) + 1
            param_type = "thickness"

        # Compute tangent of primitive quantities for this layer
        deta_layer = 0.0 + 0.0j
        dneta_layer = 0.0 + 0.0j
        drho_layer = 0.0
        dbeta_c_layer = 0.0 + 0.0j
        dea_layer = np.zeros(nfreq, dtype=np.complex128)
        deb_layer = np.zeros(nfreq, dtype=np.complex128)

        if param_type == "alpha":
            # deta/dalpha = -s_p^2 / (eta * alpha)
            deta_layer = -(s_p[layer] ** 2) / (eta[layer] * alpha[layer])
            if not np.isinf(thickness[layer]):
                dea_layer = 1j * omega * thickness[layer] * deta_layer * ea[layer, :]
        elif param_type == "beta":
            if beta[layer] > 0:
                dneta_layer = -(s_s[layer] ** 2) / (neta[layer] * beta[layer])
                dbeta_c_layer = beta_c[layer] / beta[layer]
                if not np.isinf(thickness[layer]):
                    deb_layer = (
                        1j * omega * thickness[layer] * dneta_layer * eb[layer, :]
                    )
        elif param_type == "rho":
            drho_layer = 1.0
        elif param_type == "thickness":
            if not np.isinf(thickness[layer]):
                dea_layer = 1j * omega * eta[layer] * ea[layer, :]
                deb_layer = 1j * omega * neta[layer] * eb[layer, :]

        # Compute tangent scattering matrices at affected interfaces
        # Layer `layer` appears at:
        #   interface (layer - 1): as "layer 2" (below)
        #   interface (layer):     as "layer 1" (above), if it exists

        # Pre-build zero tangent scattering matrices
        z22 = np.zeros((2, 2), dtype=np.complex128)
        dscat_Rd = [z22.copy() for _ in range(n_ifaces)]
        dscat_Ru = [z22.copy() for _ in range(n_ifaces)]
        dscat_Tu = [z22.copy() for _ in range(n_ifaces)]
        dscat_Td = [z22.copy() for _ in range(n_ifaces)]

        # Interface (layer - 1): layer is "below" (subscript 2)
        iface_below = layer - 1  # interface index
        if iface_below == 0:
            # Ocean-bottom interface
            _, _, _, _, dR, dRu, dTu, dTd = _tl_ocean_bottom(
                p,
                eta[0],
                rho[0],
                eta[1],
                neta[1],
                rho[1],
                beta_c[1],
                deta2=deta_layer,
                dneta2=dneta_layer,
                drho2=drho_layer,
                dbeta2=dbeta_c_layer,
            )
        else:
            il = iface_below  # between layers il and il+1
            _, _, _, _, dR, dRu, dTu, dTd = _tl_solid_solid(
                p,
                eta[il],
                neta[il],
                rho[il],
                beta_c[il],
                eta[il + 1],
                neta[il + 1],
                rho[il + 1],
                beta_c[il + 1],
                deta1=0.0 + 0.0j,
                dneta1=0.0 + 0.0j,
                drho1=0.0,
                dbeta1=0.0 + 0.0j,
                deta2=deta_layer,
                dneta2=dneta_layer,
                drho2=drho_layer,
                dbeta2=dbeta_c_layer,
            )
        dscat_Rd[iface_below] = dR
        dscat_Ru[iface_below] = dRu
        dscat_Tu[iface_below] = dTu
        dscat_Td[iface_below] = dTd

        # Interface (layer): layer is "above" (subscript 1), if it exists
        iface_above = layer
        if iface_above < n_ifaces:
            il = iface_above
            _, _, _, _, dR, dRu, dTu, dTd = _tl_solid_solid(
                p,
                eta[il],
                neta[il],
                rho[il],
                beta_c[il],
                eta[il + 1],
                neta[il + 1],
                rho[il + 1],
                beta_c[il + 1],
                deta1=deta_layer,
                dneta1=dneta_layer,
                drho1=drho_layer,
                dbeta1=dbeta_c_layer,
                deta2=0.0 + 0.0j,
                dneta2=0.0 + 0.0j,
                drho2=0.0,
                dbeta2=0.0 + 0.0j,
            )
            dscat_Rd[iface_above] = dR
            dscat_Ru[iface_above] = dRu
            dscat_Tu[iface_above] = dTu
            dscat_Td[iface_above] = dTd

        # Phase factor tangents: only layer `layer` is affected
        dea_all = [np.zeros(nfreq, dtype=np.complex128)] * nlayer
        deb_all = [np.zeros(nfreq, dtype=np.complex128)] * nlayer
        dea_all[layer] = dea_layer
        deb_all[layer] = deb_layer

        # --- Tangent-linear Kennett recursion ---
        dRRd = np.zeros((nfreq, 2, 2), dtype=np.complex128)

        for iface in range(n_ifaces - 1, -1, -1):
            ib = iface + 1
            # RRd at the level below (saved from forward pass)
            RRd_below = RRd_save[iface + 1]

            dRRd = _tl_kennett_step(
                scat_Rd[iface],
                scat_Ru[iface],
                scat_Tu[iface],
                scat_Td[iface],
                dscat_Rd[iface],
                dscat_Ru[iface],
                dscat_Tu[iface],
                dscat_Td[iface],
                ea[ib, :],
                eb[ib, :],
                dea_all[ib],
                deb_all[ib],
                RRd_below,
                dRRd,
            )

        # Extract PP component with ocean phase
        eaea_oc = ea[0, :] ** 2
        dR_pp = eaea_oc * dRRd[:, 0, 0]  # ocean phase is fixed (not differentiated)
        J[:, j_param] = dR_pp

    return J
