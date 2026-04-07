"""Tests for the slowness-coupled (multi-p) 9×9 interlayer MS path."""

import numpy as np
import pytest

from Kennett_Reflectivity.layer_model import LayerModel

from .interlayer_ms import (
    InterlayerMSResult9x9MultiP,
    ScattererSlab9x9,
    ScattererSlab9x9MultiP,
    interlayer_ms_reflectivity_9x9,
    interlayer_ms_reflectivity_9x9_multi_p,
)

_CD = np.complex128


# ---- Fixtures ----


@pytest.fixture
def model_4layer() -> LayerModel:
    """4-layer: ocean + 2 elastic + half-space."""
    return LayerModel.from_arrays(
        alpha=[1.5, 3.0, 4.5, 6.0],
        beta=[0.0, 1.5, 2.5, 3.5],
        rho=[1.0, 2.5, 2.8, 3.2],
        thickness=[2.0, 1.0, 1.5, np.inf],
        Q_alpha=[20000, 100, 100, 100],
        Q_beta=[1e10, 100, 100, 100],
    )


@pytest.fixture
def omega() -> complex:
    return 5.0 + 0.1j


@pytest.fixture
def kH() -> np.ndarray:
    return np.linspace(0.5, 8.0, 6)


def _broadcast_diagonal(T_single: np.ndarray, n_p: int) -> np.ndarray:
    """Embed a single ``(9, 9)`` T-matrix as a slowness-diagonal block.

    Returns ``T[p_out, p_in] = T_single · δ_{p_out, p_in}`` of shape
    ``(n_p, n_p, 9, 9)``.
    """
    T = np.zeros((n_p, n_p, 9, 9), dtype=_CD)
    for p in range(n_p):
        T[p, p] = T_single
    return T


# ===== Validation tests =====


class TestScattererSlab9x9MultiPValidation:
    """Invalid inputs raise ValueError."""

    def test_wrong_tmatrix_ndim(self, model_4layer):
        """3-D T-matrix (the single-p shape) is rejected."""
        T_bad = np.zeros((9, 9), dtype=_CD)
        with pytest.raises(ValueError, match=r"shape \(9, 9\), expected"):
            ScattererSlab9x9MultiP(
                model=model_4layer,
                scatterer_ifaces=[1],
                tmatrices={1: T_bad},
                number_densities={1: 1.0},
            )

    def test_non_square_p_block(self, model_4layer):
        """Non-square slowness block (n_p_out ≠ n_p_in) is rejected."""
        T_bad = np.zeros((4, 5, 9, 9), dtype=_CD)
        with pytest.raises(ValueError, match=r"shape \(4, 5, 9, 9\)"):
            ScattererSlab9x9MultiP(
                model=model_4layer,
                scatterer_ifaces=[1],
                tmatrices={1: T_bad},
                number_densities={1: 1.0},
            )

    def test_inconsistent_n_p_across_ifaces(self, model_4layer):
        """T-matrices with different ``n_p`` across interfaces are rejected."""
        T1 = np.zeros((4, 4, 9, 9), dtype=_CD)
        T2 = np.zeros((5, 5, 9, 9), dtype=_CD)
        with pytest.raises(ValueError, match="inconsistent"):
            ScattererSlab9x9MultiP(
                model=model_4layer,
                scatterer_ifaces=[1, 2],
                tmatrices={1: T1, 2: T2},
                number_densities={1: 1.0, 2: 1.0},
            )

    def test_iface_out_of_range(self, model_4layer):
        """Interface index out of range is rejected."""
        T = np.zeros((3, 3, 9, 9), dtype=_CD)
        with pytest.raises(ValueError, match="out of range"):
            ScattererSlab9x9MultiP(
                model=model_4layer,
                scatterer_ifaces=[99],
                tmatrices={99: T},
                number_densities={99: 1.0},
            )


# ===== Canary equivalence tests =====


class TestMultiPCanary:
    """Multi-p collapses to existing single-p when T is slowness-diagonal."""

    def test_single_iface_diagonal_matches_single_p(self, model_4layer, omega, kH):
        """T_j[p, p'] = T_single · δ(p, p') reproduces interlayer_ms_9x9."""
        n_p = len(kH)
        kx = kH.copy()
        ky = np.zeros_like(kH)

        rng = np.random.default_rng(2026_04_07)
        T_single = 1e-4 * (
            rng.standard_normal((9, 9)) + 1j * rng.standard_normal((9, 9))
        ).astype(_CD)
        T_multi = _broadcast_diagonal(T_single, n_p)

        slab_single = ScattererSlab9x9(
            model=model_4layer,
            scatterer_ifaces=[1],
            tmatrices={1: T_single},
            number_densities={1: 1.0},
        )
        res_single = interlayer_ms_reflectivity_9x9(slab_single, omega, kx, ky)

        slab_multi = ScattererSlab9x9MultiP(
            model=model_4layer,
            scatterer_ifaces=[1],
            tmatrices={1: T_multi},
            number_densities={1: 1.0},
        )
        res_multi = interlayer_ms_reflectivity_9x9_multi_p(slab_multi, omega, kx, ky)

        np.testing.assert_allclose(
            res_multi.R_background, res_single.R_background, rtol=1e-12, atol=1e-14
        )
        np.testing.assert_allclose(
            res_multi.R_total, res_single.R_total, rtol=1e-10, atol=1e-12
        )
        np.testing.assert_allclose(
            res_multi.R_born, res_single.R_born, rtol=1e-10, atol=1e-12
        )

    def test_two_ifaces_diagonal_matches_single_p(self, model_4layer, omega, kH):
        """Two-scatterer slab with diagonal T matches the single-p path."""
        n_p = len(kH)
        kx = kH.copy()
        ky = np.zeros_like(kH)

        rng = np.random.default_rng(2026_04_07_2)
        T1 = 5e-4 * (
            rng.standard_normal((9, 9)) + 1j * rng.standard_normal((9, 9))
        ).astype(_CD)
        T2 = 3e-4 * (
            rng.standard_normal((9, 9)) + 1j * rng.standard_normal((9, 9))
        ).astype(_CD)
        T1_multi = _broadcast_diagonal(T1, n_p)
        T2_multi = _broadcast_diagonal(T2, n_p)

        slab_single = ScattererSlab9x9(
            model=model_4layer,
            scatterer_ifaces=[1, 2],
            tmatrices={1: T1, 2: T2},
            number_densities={1: 0.7, 2: 1.3},
        )
        res_single = interlayer_ms_reflectivity_9x9(slab_single, omega, kx, ky)

        slab_multi = ScattererSlab9x9MultiP(
            model=model_4layer,
            scatterer_ifaces=[1, 2],
            tmatrices={1: T1_multi, 2: T2_multi},
            number_densities={1: 0.7, 2: 1.3},
        )
        res_multi = interlayer_ms_reflectivity_9x9_multi_p(slab_multi, omega, kx, ky)

        np.testing.assert_allclose(
            res_multi.R_background, res_single.R_background, rtol=1e-12, atol=1e-14
        )
        np.testing.assert_allclose(
            res_multi.R_total, res_single.R_total, rtol=1e-10, atol=1e-12
        )
        np.testing.assert_allclose(
            res_multi.R_born, res_single.R_born, rtol=1e-10, atol=1e-12
        )


# ===== Coupled-slowness behaviour =====


class TestMultiPCouplingDifferent:
    """Off-diagonal slowness coupling produces a finite, distinct answer."""

    def test_off_diagonal_coupling_changes_result(self, model_4layer, omega, kH):
        """Adding off-diagonal slowness coupling shifts R_total.

        Because ``R_total`` is extracted at ``p_emit = p_obs``, off-diagonal
        coupling ``δT[p_out ≠ p_in]`` only contributes at *second* order
        in ``δT`` (one hop to scatter out of ``p_obs``, another to scatter
        back).  We therefore use a larger off-diagonal amplitude and a
        modest threshold that robustly distinguishes the two solves while
        still being well above numerical noise.
        """
        n_p = len(kH)
        kx = kH.copy()
        ky = np.zeros_like(kH)

        rng = np.random.default_rng(2026_04_07_3)
        T_single = 1e-4 * (
            rng.standard_normal((9, 9)) + 1j * rng.standard_normal((9, 9))
        ).astype(_CD)
        T_diag = _broadcast_diagonal(T_single, n_p)

        # Add a dense off-diagonal slowness coupling, at the same scale as
        # the diagonal so the second-order effect is comfortably measurable.
        T_coupled = T_diag.copy()
        for p_out in range(n_p):
            for p_in in range(n_p):
                if p_out == p_in:
                    continue
                T_coupled[p_out, p_in] = 1e-3 * (
                    rng.standard_normal((9, 9)) + 1j * rng.standard_normal((9, 9))
                ).astype(_CD)

        slab_diag = ScattererSlab9x9MultiP(
            model=model_4layer,
            scatterer_ifaces=[1, 2],
            tmatrices={1: T_diag, 2: T_diag},
            number_densities={1: 1.0, 2: 1.0},
        )
        slab_coup = ScattererSlab9x9MultiP(
            model=model_4layer,
            scatterer_ifaces=[1, 2],
            tmatrices={1: T_coupled, 2: T_coupled},
            number_densities={1: 1.0, 2: 1.0},
        )

        res_diag = interlayer_ms_reflectivity_9x9_multi_p(slab_diag, omega, kx, ky)
        res_coup = interlayer_ms_reflectivity_9x9_multi_p(slab_coup, omega, kx, ky)

        assert np.all(np.isfinite(res_coup.R_total))
        assert np.all(np.isfinite(res_coup.R_born))
        # Distinguishable from the diagonal case.  The effect is second order
        # in the off-diagonal amplitude, so we use a second-order-scale threshold.
        rel = np.linalg.norm(res_coup.R_total - res_diag.R_total) / np.linalg.norm(
            res_diag.R_total
        )
        assert rel > 1e-4, f"Off-diagonal coupling left R_total unchanged: rel={rel}"


# ===== Result-dataclass shape contract =====


class TestMultiPResultShape:
    """The result dataclass exposes the documented shapes."""

    def test_result_shape_contract(self, model_4layer, omega, kH):
        """R arrays are (n_p,); psi arrays are (N_z, n_p_obs, n_p_state, 9)."""
        n_p = len(kH)
        kx = kH.copy()
        ky = np.zeros_like(kH)
        T = _broadcast_diagonal(1e-4 * np.eye(9, dtype=_CD), n_p)

        slab = ScattererSlab9x9MultiP(
            model=model_4layer,
            scatterer_ifaces=[1, 2],
            tmatrices={1: T, 2: T},
            number_densities={1: 1.0, 2: 1.0},
        )
        res = interlayer_ms_reflectivity_9x9_multi_p(slab, omega, kx, ky)

        assert isinstance(res, InterlayerMSResult9x9MultiP)
        assert res.R_background.shape == (n_p,)
        assert res.R_total.shape == (n_p,)
        assert res.R_born.shape == (n_p,)
        assert res.psi_exciting.shape == (2, n_p, n_p, 9)
        assert res.psi_incident.shape == (2, n_p, n_p, 9)
        # Sanity: incident is sparse (diagonal in (p_obs, p_state))
        for p_obs in range(n_p):
            for p_state in range(n_p):
                if p_obs == p_state:
                    continue
                assert np.allclose(res.psi_incident[:, p_obs, p_state, :], 0.0)


# ===== kx/ky validation =====


class TestKxKyValidation:
    """Mismatched kx/ky shapes raise."""

    def test_kx_ky_shape_mismatch(self, model_4layer, omega):
        """kx and ky with different shapes raise."""
        n_p = 4
        kx = np.linspace(0.1, 1.0, n_p)
        ky = np.zeros(n_p + 1)
        T = _broadcast_diagonal(1e-4 * np.eye(9, dtype=_CD), n_p)
        slab = ScattererSlab9x9MultiP(
            model=model_4layer,
            scatterer_ifaces=[1],
            tmatrices={1: T},
            number_densities={1: 1.0},
        )
        with pytest.raises(ValueError, match="!="):
            interlayer_ms_reflectivity_9x9_multi_p(slab, omega, kx, ky)

    def test_n_p_mismatch_with_kx(self, model_4layer, omega):
        """n_p in slab T-matrices must match kx length."""
        n_p_slab = 4
        n_p_kx = 3
        T = _broadcast_diagonal(1e-4 * np.eye(9, dtype=_CD), n_p_slab)
        slab = ScattererSlab9x9MultiP(
            model=model_4layer,
            scatterer_ifaces=[1],
            tmatrices={1: T},
            number_densities={1: 1.0},
        )
        kx = np.linspace(0.1, 1.0, n_p_kx)
        ky = np.zeros(n_p_kx)
        with pytest.raises(ValueError, match="must agree"):
            interlayer_ms_reflectivity_9x9_multi_p(slab, omega, kx, ky)
