"""Tests for the dressed T-matrix module."""

import sys
from pathlib import Path

import numpy as np

# Ensure both projects are importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_MS_ROOT = Path("/Users/tod/Desktop/MultipleScatteringCalculations")
if str(_MS_ROOT) not in sys.path:
    sys.path.insert(0, str(_MS_ROOT))

from cubic_scattering import (  # noqa: E402
    MaterialContrast,
    ReferenceMedium,
    SlabGeometry,
    compute_slab_tmatrices,
    uniform_slab_material,
)

from GlobalMatrix.dressed_tmatrix import (  # noqa: E402
    dressed_layer_tmatrix,
    self_energy_greens_9x9,
)

# ── shared test parameters ────────────────────────────────────────
REF = ReferenceMedium(alpha=3000.0, beta=1500.0, rho=2500.0)
A = 10.0  # cube half-width (m)
D = 2.0 * A
OMEGA = 2 * np.pi * 15.0 + 0.1j  # 15 Hz, small damping


def _hard_contrast() -> MaterialContrast:
    lam_sed = REF.rho * (REF.alpha**2 - 2 * REF.beta**2)
    mu_sed = REF.rho * REF.beta**2
    lam_inc = 3500.0 * (5000.0**2 - 2 * 2800.0**2)
    mu_inc = 3500.0 * 2800.0**2
    return MaterialContrast(Dlambda=lam_inc - lam_sed, Dmu=mu_inc - mu_sed, Drho=1000.0)


# ── tests ─────────────────────────────────────────────────────────


def test_zero_contrast_gives_zero():
    """T_bare = 0 when contrast is zero, so T_dressed = 0."""
    c = MaterialContrast(Dlambda=0.0, Dmu=0.0, Drho=0.0)
    T = dressed_layer_tmatrix(A, REF, c, OMEGA, n_rings=3)
    np.testing.assert_allclose(T, 0.0, atol=1e-30)


def test_no_neighbors_gives_bare():
    """With n_rings=0, G_self = 0 so T_dressed = T_bare."""
    c = _hard_contrast()
    T_dressed = dressed_layer_tmatrix(A, REF, c, OMEGA, n_rings=0)

    geom = SlabGeometry(M=1, N_z=1, a=A)
    mat = uniform_slab_material(geom, REF, c)
    T_bare = compute_slab_tmatrices(geom, mat, OMEGA)[0, 0, 0]  # type: ignore[arg-type]

    np.testing.assert_allclose(T_dressed, T_bare, rtol=1e-12)


def test_weak_contrast_small_correction():
    """For weak contrast, dressed ≈ bare (correction is O(contrast²))."""
    c_hard = _hard_contrast()
    scale = 0.01
    c_weak = MaterialContrast(
        Dlambda=scale * c_hard.Dlambda,
        Dmu=scale * c_hard.Dmu,
        Drho=scale * c_hard.Drho,
    )

    T_dressed = dressed_layer_tmatrix(A, REF, c_weak, OMEGA, n_rings=10)

    geom = SlabGeometry(M=1, N_z=1, a=A)
    mat = uniform_slab_material(geom, REF, c_weak)
    T_bare = compute_slab_tmatrices(geom, mat, OMEGA)[0, 0, 0]  # type: ignore[arg-type]

    rel_diff = np.linalg.norm(T_dressed - T_bare) / np.linalg.norm(T_bare)
    assert rel_diff < 0.1, f"Correction {rel_diff:.4f} too large for weak contrast"


def test_convergence_n_rings():
    """G_self at n_rings=20 vs 30 should agree to < 5%."""
    G20 = self_energy_greens_9x9(D, OMEGA, REF, n_rings=20)
    G30 = self_energy_greens_9x9(D, OMEGA, REF, n_rings=30)

    rel_diff = np.linalg.norm(G30 - G20) / np.linalg.norm(G30)
    assert rel_diff < 0.05, f"Convergence failed: rel diff = {rel_diff:.4e}"


def test_output_shape_and_dtype():
    """T_dressed is (9, 9) complex."""
    c = _hard_contrast()
    T = dressed_layer_tmatrix(A, REF, c, OMEGA, n_rings=3)
    assert T.shape == (9, 9)
    assert np.iscomplexobj(T)
