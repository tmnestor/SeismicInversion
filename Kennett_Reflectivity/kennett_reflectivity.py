"""Kennett recursive reflectivity algorithm for stratified media.

Faithful translation of Kennet_Reflex subroutine from kennetslo.f.
Vectorised over frequency using NumPy broadcasting for batch 2x2 operations.
"""

from __future__ import annotations

import numpy as np

from .layer_model import LayerModel, vertical_slowness
from .scattering_matrices import ocean_bottom_interface, solid_solid_interface

__all__ = ["kennett_reflectivity", "inv2x2", "batch_inv2x2", "batch_matmul"]


def inv2x2(M: np.ndarray) -> np.ndarray:
    """
    Analytical inverse of a single 2x2 matrix.

    For [[a,b],[c,d]]: inv = 1/(ad-bc) * [[d,-b],[-c,a]]
    """
    a, b = M[0, 0], M[0, 1]
    c, d = M[1, 0], M[1, 1]
    det = a * d - b * c
    return np.array([[d, -b], [-c, a]], dtype=M.dtype) / det


def batch_inv2x2(M: np.ndarray) -> np.ndarray:
    """
    Vectorised analytical inverse of a batch of 2x2 matrices.

    Parameters
    ----------
    M : np.ndarray, shape (nfreq, 2, 2)

    Returns
    -------
    np.ndarray, shape (nfreq, 2, 2)
    """
    a = M[:, 0, 0]
    b = M[:, 0, 1]
    c = M[:, 1, 0]
    d = M[:, 1, 1]
    det = a * d - b * c
    inv = np.empty_like(M)
    inv[:, 0, 0] = d / det
    inv[:, 0, 1] = -b / det
    inv[:, 1, 0] = -c / det
    inv[:, 1, 1] = a / det
    return inv


def batch_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Batch matrix multiply for 2x2 matrices.

    Supports shapes:
        (2,2) @ (nfreq,2,2) -> (nfreq,2,2)
        (nfreq,2,2) @ (2,2) -> (nfreq,2,2)
        (nfreq,2,2) @ (nfreq,2,2) -> (nfreq,2,2)
    """
    if A.ndim == 2 and B.ndim == 3:
        return np.einsum("ij,wjk->wik", A, B)
    elif A.ndim == 3 and B.ndim == 2:
        return np.einsum("wij,jk->wik", A, B)
    elif A.ndim == 3 and B.ndim == 3:
        return np.einsum("wij,wjk->wik", A, B)
    else:
        raise ValueError(f"Unsupported shapes: {A.shape}, {B.shape}")


def kennett_reflectivity(
    model: LayerModel,
    p: float,
    omega: np.ndarray,
    free_surface: bool = False,
) -> np.ndarray:
    """Compute plane-wave reflectivity using Kennett's addition formula.

    Faithful translation of Kennet_Reflex from kennetslo.f, vectorised
    over frequency using NumPy batch 2x2 operations.

    Optionally includes free surface reflections (surface multiples)
    via the ocean layer reverberation operator.

    Parameters
    ----------
    model : LayerModel
        The stratified elastic model.
    p : float
        Horizontal slowness (ray parameter).
    omega : np.ndarray
        Angular frequencies, shape (nfreq,). Should NOT include DC (omega=0).
    free_surface : bool
        If True, include free surface reflections at the ocean surface.
        The free surface reflection coefficient for an acoustic medium
        (pressure-release boundary) is R_fs = -1. The ocean layer then
        acts as a reverberant waveguide, generating surface multiples.
        Default: False (original Fortran behaviour).

    Returns
    -------
    np.ndarray
        Complex reflectivity (PP component) at each frequency, shape (nfreq,).

    Algorithm
    ---------
    Upward sweep from half-space to ocean bottom using Kennett's addition formula:
        1. RRd = 0 at half-space (radiation condition)
        2. At bottom interface: RRd = Rd (no phase needed for half-space)
        3. For each interface il from bottom to ocean-bottom:
           - MT = E * RRd * E (phase-shifted cumulative reflection)
           - U = (I - Ru * MT)^{-1}
           - RRd = Rd + Tu * MT * U * Td
        4. At ocean bottom: extract PP component and multiply by ocean phase

    If free_surface=True, an additional reverberation step is applied:

        The ocean layer sits between the free surface (top, R_fs = -1)
        and the sub-ocean reflectivity RRd (bottom). A P-wave travelling
        in the ocean reverberates between these two boundaries.

        Let E_oc = exp(iω η_ocean h_ocean) be the one-way ocean phase.
        The PP reflectivity seen at the ocean bottom, including all
        surface multiples, is given by Kennett's addition formula
        applied to the ocean layer:

            R_total = RRd_PP
                    + Td_ob · E²_oc · R_fs · (1 - RRd_PP · E²_oc · R_fs)⁻¹ · Tu_ob

        where:
            RRd_PP  = sub-ocean cumulative PP reflectivity (from upward sweep)
            Td_ob   = ocean-bottom downgoing PP transmission (ocean → sediment direction)
            Tu_ob   = ocean-bottom upgoing PP transmission (sediment → ocean direction)
            R_fs    = -1 (pressure-release free surface)
            E_oc    = exp(iω η_ocean h_ocean) (one-way P-wave phase through ocean)

        However, for an ocean-bottom source/receiver configuration, the
        response observed at the ocean surface is simpler. We use Kennett's
        general addition formula to combine the free surface with the
        ocean-bottom reflectivity:

            R_total(ω) = E²_oc · RRd_PP / (1 - R_fs · E²_oc · RRd_PP)

        This is the scalar form because:
        - The ocean is acoustic (no S-waves), so only PP propagates
        - R_fs = -1 is a scalar for the pressure-release surface
        - The ocean phase E²_oc is scalar (one P-wave mode)

        The denominator (1 + E²_oc · RRd_PP) generates the infinite
        reverberation series: primary + first surface multiple +
        second surface multiple + ... Each term adds another round
        trip through the ocean layer reflected off the free surface.
    """
    nfreq = len(omega)
    nlayer = model.n_layers

    # Complex slownesses (Fortran: A, B, BETA arrays)
    s_p = model.complex_slowness_p()  # shape (nlayer,)
    s_s = model.complex_slowness_s()  # shape (nlayer,)
    beta_c = model.complex_velocity_s()  # shape (nlayer,), complex velocity

    # Vertical slownesses for all layers (Fortran: ETA, NETA)
    cp = complex(p)
    eta = np.array(
        [vertical_slowness(s_p[i], cp) for i in range(nlayer)], dtype=np.complex128
    )
    neta = np.zeros(nlayer, dtype=np.complex128)
    neta[0] = 0.0  # acoustic ocean: NETA(1) = C0
    for i in range(1, nlayer):
        neta[i] = vertical_slowness(s_s[i], cp)

    # Compute phase shift factors for all layers and frequencies
    # ea[i, w] = exp(i * omega[w] * eta[i] * thickness[i])
    # eb[i, w] = exp(i * omega[w] * neta[i] * thickness[i])
    ea = np.ones((nlayer, nfreq), dtype=np.complex128)
    eb = np.ones((nlayer, nfreq), dtype=np.complex128)
    for i in range(nlayer):
        if not np.isinf(model.thickness[i]):
            ea[i, :] = np.exp(1j * omega * eta[i] * model.thickness[i])
            eb[i, :] = np.exp(1j * omega * neta[i] * model.thickness[i])

    # Precompute scattering coefficients at all interfaces (frequency-independent)
    # Interface 0: ocean-bottom (acoustic-elastic)
    # Interfaces 1..nlayer-2: solid-solid
    scat = []
    # Interface 0: ocean (layer 0) / sediment (layer 1)
    scat.append(
        ocean_bottom_interface(
            p=p,
            eta1=eta[0],
            rho1=model.rho[0],
            eta2=eta[1],
            neta2=neta[1],
            rho2=model.rho[1],
            beta2=beta_c[1],
        )
    )
    # Interfaces 1..nlayer-2: solid-solid
    for il in range(1, nlayer - 1):
        scat.append(
            solid_solid_interface(
                p=p,
                eta1=eta[il],
                neta1=neta[il],
                rho1=model.rho[il],
                beta1=beta_c[il],
                eta2=eta[il + 1],
                neta2=neta[il + 1],
                rho2=model.rho[il + 1],
                beta2=beta_c[il + 1],
            )
        )

    # ===== KENNETT UPWARD SWEEP =====
    # Initialise: RRd = 0 at half-space (radiation condition)
    RRd = np.zeros((nfreq, 2, 2), dtype=np.complex128)
    I2 = np.eye(2, dtype=np.complex128)

    # Number of interfaces = nlayer - 1
    n_interfaces = nlayer - 1  # scat[0] to scat[n_interfaces-1]

    # Start from bottom interface and sweep upward
    # scat[n_interfaces-1] is the bottom interface (e.g., upper mantle / half-space)
    # scat[0] is the ocean bottom
    for iface in range(n_interfaces - 1, -1, -1):
        # iface indexes the interface: between layers iface and iface+1
        i_below = iface + 1  # layer below this interface

        coeff = scat[iface]

        # Phase shift through the layer below (two-way travel)
        # eaea = ea[i_below]^2, ebeb = eb[i_below]^2, eaeb = ea*eb
        eaea = ea[i_below, :] ** 2  # shape (nfreq,)
        ebeb = eb[i_below, :] ** 2
        eaeb = ea[i_below, :] * eb[i_below, :]

        # MT = E * RRd * E (diagonal phase scaling of 2x2 matrix)
        MT = np.empty((nfreq, 2, 2), dtype=np.complex128)
        MT[:, 0, 0] = eaea * RRd[:, 0, 0]
        MT[:, 0, 1] = eaeb * RRd[:, 0, 1]
        MT[:, 1, 0] = eaeb * RRd[:, 1, 0]
        MT[:, 1, 1] = ebeb * RRd[:, 1, 1]

        # U = (I - Ru * MT)^{-1}
        # Compute Ru @ MT for each frequency (Ru is 2x2, MT is batch)
        RuMT = batch_matmul(coeff.Ru, MT)  # (nfreq, 2, 2)
        I_minus_RuMT = I2[np.newaxis, :, :] - RuMT  # (nfreq, 2, 2)
        U = batch_inv2x2(I_minus_RuMT)  # (nfreq, 2, 2)

        # RRd_new = Rd + Tu * MT * U * Td
        MTU = batch_matmul(MT, U)  # MT @ U
        TuMTU = batch_matmul(coeff.Tu, MTU)  # Tu @ (MT @ U)
        TuMTUTd = batch_matmul(TuMTU, coeff.Td)  # Tu @ MT @ U @ Td
        RRd = coeff.Rd[np.newaxis, :, :] + TuMTUTd  # broadcast Rd

    # ===== EXTRACT PP REFLECTIVITY =====

    # RRd[:, 0, 0] is the cumulative PP reflectivity of the entire
    # sub-ocean stack, as seen from immediately below the ocean layer.
    # This includes all internal multiples within the solid layers
    # but NO free surface effects.
    RRd_PP = RRd[:, 0, 0]  # shape (nfreq,)

    # Two-way P-wave phase through the ocean layer:
    # E²_oc = exp(2iω η_ocean h_ocean)
    # This accounts for the source-to-ocean-bottom-and-back travel time.
    eaea_ocean = ea[0, :] ** 2  # shape (nfreq,)

    if not free_surface:
        # ----- Original behaviour (no free surface) -----
        # Simple phase-shifted reflectivity: the wave travels down
        # through the ocean, reflects off the sub-ocean stack, and
        # travels back up. No interaction with the ocean surface.
        R = eaea_ocean * RRd_PP

    else:
        # ----- Include free surface reflections -----
        #
        # Physics: the ocean layer is bounded by the free surface
        # at the top (R_fs = -1 for a pressure-release boundary)
        # and the sub-ocean reflectivity RRd_PP at the bottom.
        # A P-wave reverberates between these two boundaries,
        # generating surface multiples (water-column reverberations).
        #
        # The free surface reflection coefficient for an acoustic
        # half-space with a pressure-release (zero pressure) boundary:
        #
        #   R_fs = -1
        #
        # This is exact for all angles of incidence in a fluid.
        # The sign convention is: an upgoing wave hitting the free
        # surface produces a downgoing wave with opposite polarity.
        R_fs = -1.0

        # Kennett's addition formula for the ocean layer gives the
        # total reflectivity including all surface multiples:
        #
        #   R_total = E²_oc · RRd_PP / (1 - R_fs · E²_oc · RRd_PP)
        #
        # Since R_fs = -1, the denominator becomes:
        #
        #   1 - (-1) · E²_oc · RRd_PP = 1 + E²_oc · RRd_PP
        #
        # Expanding as a geometric series:
        #
        #   R_total = E²_oc · RRd_PP · Σₙ (-E²_oc · RRd_PP)ⁿ
        #           = E²_oc · RRd_PP              (primary)
        #           - E⁴_oc · RRd²_PP             (1st surface multiple)
        #           + E⁶_oc · RRd³_PP             (2nd surface multiple)
        #           - ...
        #
        # Each successive term adds one more round trip through the
        # ocean layer (extra E²_oc phase) and one more reflection
        # off the sub-ocean stack (extra RRd_PP factor), with
        # alternating sign from each free surface bounce (R_fs = -1).

        # Numerator: wave travels through ocean, reflects, returns
        numerator = eaea_ocean * RRd_PP

        # Denominator: reverberation operator
        # 1 + E²_oc · RRd_PP  (since R_fs = -1)
        denominator = 1.0 + eaea_ocean * RRd_PP

        R = numerator / denominator

    return R
