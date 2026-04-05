"""Kennett recursive reflectivity — batched NumPy implementation.

Batches the Kennett recursion across ALL slowness samples simultaneously
using 4D NumPy arrays of shape (np_slow, nfreq, 2, 2) in float64.
This eliminates the Python-level loop over slowness and the need for
multiprocessing, while maintaining full double-precision accuracy.

The GPU (MPS/CUDA) acceleration is applied in kennett_gather_gpu.py
to the Bessel-weighted wavenumber summation, which is the true
computational bottleneck and is float32-safe (no underflow risk).

Performance scaling:
    CPU serial:       np_slow × kennett_reflectivity calls
    CPU multiprocess: np_slow / n_workers calls
    This module:      1 batched call (vectorised over np_slow × nfreq)

References
----------
Kennett, B. L. N. (1983). Seismic Wave Propagation in Stratified Media.
Aki, K. & Richards, P. G. (1980). Quantitative Seismology.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from tqdm import tqdm

from .layer_model import LayerModel, vertical_slowness
from .scattering_matrices import ocean_bottom_interface, solid_solid_interface

__all__ = ["kennett_reflectivity_batch", "get_device"]

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Select the best available PyTorch device.

    Priority: MPS (Apple Silicon) > CUDA > CPU.

    Returns
    -------
    torch.device
        The selected device.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _batch_inv2x2(M: np.ndarray) -> np.ndarray:
    """Analytical inverse of batched 2×2 complex matrices (NumPy).

    Parameters
    ----------
    M : np.ndarray, shape (..., 2, 2)
        Batch of 2×2 complex matrices.

    Returns
    -------
    np.ndarray, shape (..., 2, 2)
        Element-wise inverse of each 2×2 matrix.

    Notes
    -----
    Uses the analytical formula: inv([[a,b],[c,d]]) = [[d,-b],[-c,a]] / (ad-bc).
    Avoids np.linalg.inv overhead for the 2×2 case.
    """
    a = M[..., 0, 0]
    b = M[..., 0, 1]
    c = M[..., 1, 0]
    d = M[..., 1, 1]
    det = a * d - b * c

    inv = np.empty_like(M)
    inv[..., 0, 0] = d / det
    inv[..., 0, 1] = -b / det
    inv[..., 1, 0] = -c / det
    inv[..., 1, 1] = a / det
    return inv


def kennett_reflectivity_batch(
    model: LayerModel,
    p_samples: np.ndarray,
    omega: np.ndarray,
    free_surface: bool = False,
) -> np.ndarray:
    """Compute reflectivity for ALL slowness samples simultaneously.

    Batched NumPy implementation that processes all slowness values in
    a single vectorised pass. All computation is in float64 for full
    double-precision accuracy (critical for evanescent phase factors
    that underflow in float32).

    Parameters
    ----------
    model : LayerModel
        The stratified elastic model.
    p_samples : np.ndarray
        Horizontal slowness values, shape (np_slow,).
    omega : np.ndarray
        Angular frequencies, shape (nfreq,). Should NOT include DC (ω=0).
    free_surface : bool
        If True, include free surface reflections at the ocean surface.

    Returns
    -------
    np.ndarray
        Complex PP reflectivity, shape (np_slow, nfreq), dtype complex128.

    Algorithm
    ---------
    1. Precompute vertical slownesses η(p) for each layer/slowness.
    2. Precompute scattering matrices at each interface for each p.
    3. Precompute phase factors ea, eb for each (layer, slowness, freq).
    4. Run Kennett upward sweep with RRd of shape (np_slow, nfreq, 2, 2).
    5. Extract PP component and apply ocean phase + free surface operator.
    """
    np_slow = len(p_samples)
    nfreq = len(omega)
    nlayer = model.n_layers

    logger.info(
        f"Batched reflectivity: {np_slow} slowness × {nfreq} frequencies "
        f"(float64, vectorised NumPy)"
    )

    # ===== PRECOMPUTE LAYER PROPERTIES =====
    s_p = model.complex_slowness_p()  # (nlayer,)
    s_s = model.complex_slowness_s()  # (nlayer,)
    beta_c = model.complex_velocity_s()  # (nlayer,)

    # Vertical slownesses: eta[i, jp], neta[i, jp]
    eta = np.zeros((nlayer, np_slow), dtype=np.complex128)
    neta = np.zeros((nlayer, np_slow), dtype=np.complex128)
    for i in tqdm(range(nlayer), desc="Vertical slownesses", unit="layer"):
        for jp in range(np_slow):
            cp = complex(p_samples[jp])
            eta[i, jp] = vertical_slowness(s_p[i], cp)
            if i > 0:  # layer 0 is acoustic
                neta[i, jp] = vertical_slowness(s_s[i], cp)

    # ===== PRECOMPUTE SCATTERING MATRICES =====
    n_interfaces = nlayer - 1

    Rd_all = np.zeros((n_interfaces, np_slow, 2, 2), dtype=np.complex128)
    Ru_all = np.zeros((n_interfaces, np_slow, 2, 2), dtype=np.complex128)
    Tu_all = np.zeros((n_interfaces, np_slow, 2, 2), dtype=np.complex128)
    Td_all = np.zeros((n_interfaces, np_slow, 2, 2), dtype=np.complex128)

    for jp in tqdm(range(np_slow), desc="Scattering matrices", unit="p"):
        p_val = float(p_samples[jp])

        # Interface 0: ocean / sediment
        coeff = ocean_bottom_interface(
            p=p_val,
            eta1=eta[0, jp],
            rho1=model.rho[0],
            eta2=eta[1, jp],
            neta2=neta[1, jp],
            rho2=model.rho[1],
            beta2=beta_c[1],
        )
        Rd_all[0, jp] = coeff.Rd
        Ru_all[0, jp] = coeff.Ru
        Tu_all[0, jp] = coeff.Tu
        Td_all[0, jp] = coeff.Td

        # Interfaces 1..nlayer-2: solid-solid
        for il in range(1, nlayer - 1):
            coeff = solid_solid_interface(
                p=p_val,
                eta1=eta[il, jp],
                neta1=neta[il, jp],
                rho1=model.rho[il],
                beta1=beta_c[il],
                eta2=eta[il + 1, jp],
                neta2=neta[il + 1, jp],
                rho2=model.rho[il + 1],
                beta2=beta_c[il + 1],
            )
            Rd_all[il, jp] = coeff.Rd
            Ru_all[il, jp] = coeff.Ru
            Tu_all[il, jp] = coeff.Tu
            Td_all[il, jp] = coeff.Td

    # ===== PRECOMPUTE PHASE FACTORS (float64 — no underflow) =====
    # ea[i, jp, w] = exp(iω · η[i,jp] · h[i]), shape (nlayer, np_slow, nfreq)
    omega_np = np.asarray(omega, dtype=np.complex128)

    ea = np.ones((nlayer, np_slow, nfreq), dtype=np.complex128)
    eb = np.ones((nlayer, np_slow, nfreq), dtype=np.complex128)
    for i in range(nlayer):
        if not np.isinf(model.thickness[i]):
            phase_p = (
                1j
                * eta[i, :, np.newaxis]
                * omega_np[np.newaxis, :]
                * model.thickness[i]
            )
            ea[i] = np.exp(phase_p)
            phase_s = (
                1j
                * neta[i, :, np.newaxis]
                * omega_np[np.newaxis, :]
                * model.thickness[i]
            )
            eb[i] = np.exp(phase_s)

    # ===== KENNETT UPWARD SWEEP (batched NumPy, float64) =====
    # RRd shape: (np_slow, nfreq, 2, 2)
    RRd = np.zeros((np_slow, nfreq, 2, 2), dtype=np.complex128)

    iface_range = range(n_interfaces - 1, -1, -1)
    for iface in tqdm(iface_range, desc="Kennett recursion", unit="iface"):
        i_below = iface + 1

        # Scattering coefficients: (np_slow, 2, 2)
        Rd_if = Rd_all[iface]
        Ru_if = Ru_all[iface]
        Tu_if = Tu_all[iface]
        Td_if = Td_all[iface]

        # Phase factors for layer below: (np_slow, nfreq)
        ea_below = ea[i_below]
        eb_below = eb[i_below]

        # Two-way phase products: (np_slow, nfreq)
        eaea = ea_below**2
        ebeb = eb_below**2
        eaeb = ea_below * eb_below

        # MT = E · RRd · E: phase-shifted cumulative reflection
        MT = np.empty_like(RRd)
        MT[:, :, 0, 0] = eaea * RRd[:, :, 0, 0]
        MT[:, :, 0, 1] = eaeb * RRd[:, :, 0, 1]
        MT[:, :, 1, 0] = eaeb * RRd[:, :, 1, 0]
        MT[:, :, 1, 1] = ebeb * RRd[:, :, 1, 1]

        # U = (I - Ru · MT)^{-1}
        # Ru_if[:, None, :, :] @ MT -> (np_slow, nfreq, 2, 2)
        RuMT = np.matmul(Ru_if[:, np.newaxis, :, :], MT)
        I_minus_RuMT = np.eye(2, dtype=np.complex128) - RuMT
        U = _batch_inv2x2(I_minus_RuMT)

        # RRd_new = Rd + Tu · MT · U · Td
        MTU = np.matmul(MT, U)
        TuMTU = np.matmul(Tu_if[:, np.newaxis, :, :], MTU)
        TuMTUTd = np.matmul(TuMTU, Td_if[:, np.newaxis, :, :])

        RRd = Rd_if[:, np.newaxis, :, :] + TuMTUTd

    # ===== EXTRACT PP REFLECTIVITY =====
    RRd_PP = RRd[:, :, 0, 0]  # (np_slow, nfreq)

    # Two-way P-wave phase through ocean
    eaea_ocean = ea[0] ** 2  # (np_slow, nfreq)

    if not free_surface:
        R = eaea_ocean * RRd_PP
    else:
        # Free surface reverberation operator:
        # R_total = E²_oc · RRd_PP / (1 + E²_oc · RRd_PP)
        numerator = eaea_ocean * RRd_PP
        denominator = 1.0 + eaea_ocean * RRd_PP
        R = numerator / denominator

    logger.info("Batched reflectivity computation complete")
    return R
