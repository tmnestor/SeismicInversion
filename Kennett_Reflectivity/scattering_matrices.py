"""Interfacial P-SV reflection and transmission coefficients.

Faithful line-by-line translation of ScatMat and OBSMat subroutines
from kennetslo.f. Implements modified interfacial scattering matrices
following Aki & Richards (1980) pp. 149-151.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ScatteringCoefficients",
    "solid_solid_interface",
    "ocean_bottom_interface",
]


@dataclass
class ScatteringCoefficients:
    """
    2x2 P-SV scattering coefficients at an interface.

    Each matrix is 2x2 complex: index 0 = P-wave, index 1 = S-wave.
    """

    Rd: np.ndarray  # 2x2, downgoing reflection
    Ru: np.ndarray  # 2x2, upgoing reflection
    Tu: np.ndarray  # 2x2, upgoing transmission
    Td: np.ndarray  # 2x2, downgoing transmission

    def __post_init__(self):
        for name in ("Rd", "Ru", "Tu", "Td"):
            arr = np.asarray(getattr(self, name), dtype=np.complex128)
            if arr.shape != (2, 2):
                raise ValueError(f"{name} must be 2x2, got {arr.shape}")
            setattr(self, name, arr)


def solid_solid_interface(
    p: float,
    eta1: complex,
    neta1: complex,
    rho1: float,
    beta1: complex,
    eta2: complex,
    neta2: complex,
    rho2: float,
    beta2: complex,
) -> ScatteringCoefficients:
    """
    Compute modified P-SV scattering coefficients at a solid-solid interface.

    Line-by-line translation of ScatMat from kennetslo.f.

    Parameters
    ----------
    p : float
        Horizontal slowness (ray parameter).
    eta1, eta2 : complex
        Vertical P-wave slowness in layers 1, 2.
    neta1, neta2 : complex
        Vertical S-wave slowness in layers 1, 2.
    rho1, rho2 : float
        Densities.
    beta1, beta2 : complex
        Complex S-wave velocity (= 1/complex_s_slowness, NOT real velocity).

    Returns
    -------
    ScatteringCoefficients
        Modified scattering matrices Rd, Ru, Tu, Td.

    Notes
    -----
    The Fortran uses 'modified' coefficients incorporating sqrt(eta*rho) factors
    so that the Kennett recursion preserves unitarity automatically.
    Tu = Td^T (reciprocity).
    """
    # Fortran: rtrho, rteta, rtneta, rtza, rtzb
    rtrho1 = np.sqrt(rho1)
    rtrho2 = np.sqrt(rho2)
    rteta1 = np.sqrt(eta1)
    rteta2 = np.sqrt(eta2)
    rtneta1 = np.sqrt(neta1)
    rtneta2 = np.sqrt(neta2)
    rtza1 = rteta1 * rtrho1
    rtza2 = rteta2 * rtrho2
    rtzb1 = rtneta1 * rtrho1
    rtzb2 = rtneta2 * rtrho2

    psq = complex(p * p)
    crho1 = complex(rho1)
    crho2 = complex(rho2)
    drho = crho2 - crho1
    dmu = crho2 * beta2 * beta2 - crho1 * beta1 * beta1

    d = 2.0 * dmu
    psqd = psq * d
    a = drho - psqd
    b = crho2 - psqd
    c = crho1 + psqd

    # Fortran E,F,G,H (before division by Det)
    E = b * eta1 + c * eta2
    F = b * neta1 + c * neta2
    G = a - d * eta1 * neta2
    H = a - d * eta2 * neta1

    Det = E * F + G * H * psq

    # Divide E,F,G,H by Det (Fortran: E=E/Det etc.)
    # Det itself is NOT modified — used again for S_val, V_val
    E = E / Det
    F = F / Det
    G = G / Det
    H = H / Det

    # Intermediate quantities (Fortran Q,R,S,T,U,V)
    Q_val = (b * eta1 - c * eta2) * F
    R_val = (a + d * eta1 * neta2) * H * psq
    S_val = (a * b + c * d * eta2 * neta2) * p / Det
    T_val = (b * neta1 - c * neta2) * E
    U_val = (a + d * eta2 * neta1) * G * psq
    V_val = (a * c + b * d * eta1 * neta1) * p / Det

    m2ci = complex(0.0, -2.0)  # Fortran m2Ci = (0.0, -2.0)

    # Rd: downgoing reflection (Fortran Rd11=Q-R, Rd12=Rd21=-2i*rt*S, Rd22=T-U)
    Rd = np.array(
        [
            [Q_val - R_val, m2ci * rteta1 * rtneta1 * S_val],
            [m2ci * rteta1 * rtneta1 * S_val, T_val - U_val],
        ],
        dtype=np.complex128,
    )

    # Td: downgoing transmission
    Td = np.array(
        [
            [2 * rtza1 * rtza2 * F, m2ci * rtzb1 * rtza2 * G * p],
            [m2ci * rtza1 * rtzb2 * H * p, 2 * rtzb1 * rtzb2 * E],
        ],
        dtype=np.complex128,
    )

    # Tu = Td^T (reciprocity: Tu11=Td11, Tu12=Td21, Tu21=Td12, Tu22=Td22)
    Tu = Td.T.copy()

    # Ru: upgoing reflection
    Ru = np.array(
        [
            [-(Q_val + U_val), m2ci * rteta2 * rtneta2 * V_val],
            [m2ci * rteta2 * rtneta2 * V_val, -(T_val + R_val)],
        ],
        dtype=np.complex128,
    )

    return ScatteringCoefficients(Rd=Rd, Ru=Ru, Tu=Tu, Td=Td)


def ocean_bottom_interface(
    p: float,
    eta1: complex,
    rho1: float,
    eta2: complex,
    neta2: complex,
    rho2: float,
    beta2: complex,
) -> ScatteringCoefficients:
    """
    Compute scattering coefficients at ocean-bottom (acoustic-elastic) interface.

    Line-by-line translation of OBSMat from kennetslo.f.
    Layer 1 (ocean) is acoustic: beta1=0, neta1=0.

    Parameters
    ----------
    p : float
        Horizontal slowness.
    eta1 : complex
        Vertical P-wave slowness in ocean.
    rho1 : float
        Ocean density.
    eta2 : complex
        Vertical P-wave slowness in elastic layer.
    neta2 : complex
        Vertical S-wave slowness in elastic layer.
    rho2 : float
        Density of elastic layer.
    beta2 : complex
        Complex S-wave velocity (= 1/complex_s_slowness).

    Returns
    -------
    ScatteringCoefficients
        Scattering matrices with acoustic-elastic structure.
        Rd has only [0,0] nonzero; Td has column 1 zero; Tu has row 1 zero.

    Notes
    -----
    Fortran output mapping:
        PdPu -> Rd[0,0]    PdPd -> Td[0,0]    PdSd -> Td[1,0]
        PuPu -> Tu[0,0]    PuPd -> Ru[0,0]    PuSd -> Ru[1,0]
        SuPu -> Tu[0,1]    SuPd -> Ru[0,1]    SuSd -> Ru[1,1]
    """
    rtrho1 = np.sqrt(rho1)
    rtrho2 = np.sqrt(rho2)
    rteta1 = np.sqrt(eta1)
    rteta2 = np.sqrt(eta2)
    rtneta2 = np.sqrt(neta2)
    rtza1 = rtrho1 * rteta1
    rtza2 = rtrho2 * rteta2
    rtzb2 = rtrho2 * rtneta2

    psq = p * p
    crho1 = complex(rho1)
    crho2 = complex(rho2)
    drho = crho2 - crho1
    dmu = crho2 * beta2 * beta2  # acoustic: no beta1 term

    d = 2.0 * dmu
    psqd = psq * d
    a = drho - psqd
    b = crho2 - psqd
    c = crho1 + psqd

    # Fortran OBSMat: special forms for acoustic-elastic
    # F = b (not b*neta1 + c*neta2, since neta1=0 in acoustic)
    # H = -d*eta2 (not a - d*eta2*neta1, since neta1=0)
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
    T6 = b * d * eta1 * p / Det  # uses original Det (not modified)

    mci = complex(0.0, -1.0)  # Fortran mCi = (0.0, -1.0)

    # Fortran output assignments
    PdPu = T1 - T2  # Rd11
    PdPd = 2.0 * rtza1 * rtza2 * F  # Td11
    PdSd = 2.0 * mci * rtza1 * rtzb2 * H * p  # Td21
    PuPu = PdPd  # Tu11 = Td11
    SuPu = PdSd  # Tu12 = Td21
    PuPd = -(T1 + T5)  # Ru11
    PuSd = 2.0 * mci * rteta2 * rtneta2 * T6  # Ru21
    SuPd = PuSd  # Ru12 = Ru21
    SuSd = -(T2 + T4)  # Ru22

    # Build 2x2 matrices with correct acoustic-elastic structure
    # Unset elements are zero (no downgoing S from ocean, no upgoing S to ocean)
    Rd = np.array(
        [
            [PdPu, 0.0 + 0j],
            [0.0 + 0j, 0.0 + 0j],
        ],
        dtype=np.complex128,
    )

    Td = np.array(
        [
            [PdPd, 0.0 + 0j],
            [PdSd, 0.0 + 0j],
        ],
        dtype=np.complex128,
    )

    Tu = np.array(
        [
            [PuPu, SuPu],
            [0.0 + 0j, 0.0 + 0j],
        ],
        dtype=np.complex128,
    )

    Ru = np.array(
        [
            [PuPd, SuPd],
            [PuSd, SuSd],
        ],
        dtype=np.complex128,
    )

    return ScatteringCoefficients(Rd=Rd, Ru=Ru, Tu=Tu, Td=Td)
