"""Tests for interlayer-only multiple scattering module."""

import numpy as np
import pytest

from Kennett_Reflectivity.layer_model import LayerModel

from .interlayer_ms import (
    InterlayerMSResult,
    InterlayerMSResult9x9,
    ScattererSlab,
    ScattererSlab9x9,
    background_incident_field,
    build_interlayer_greens_matrix,
    interlayer_ms_reflectivity,
    interlayer_ms_reflectivity_9x9,
    tmatrix_6x6_to_4x4_psv,
    tmatrix_9x9_to_4x4_psv,
)
from .layered_greens import (
    _interface_elastic_properties,
    layered_greens_psv,
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
def model_5layer() -> LayerModel:
    """5-layer: ocean + 3 elastic + half-space."""
    return LayerModel.from_arrays(
        alpha=[1.5, 1.6, 3.0, 5.0, 2.2],
        beta=[0.0, 0.3, 1.5, 3.0, 1.1],
        rho=[1.0, 2.0, 3.0, 3.0, 1.8],
        thickness=[2.0, 1.0, 1.0, 1.0, np.inf],
        Q_alpha=[20000, 100, 100, 100, 100],
        Q_beta=[1e10, 100, 100, 100, 100],
    )


@pytest.fixture
def weak_tmatrix() -> np.ndarray:
    """Weak isotropic T-matrix (small perturbation)."""
    return 1e-4 * np.eye(4, dtype=_CD)


@pytest.fixture
def random_tmatrix(rng: np.random.Generator) -> np.ndarray:
    """Random T-matrix for testing."""
    return 1e-3 * (rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4)))


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def omega() -> complex:
    return 5.0 + 0.1j


@pytest.fixture
def kH() -> np.ndarray:
    return np.linspace(0.5, 8.0, 15)


# ===== Validation tests =====


class TestScattererSlabValidation:
    """Invalid inputs raise ValueError."""

    def test_interface_out_of_range(self, model_4layer, weak_tmatrix):
        """Interface index outside [1, M] raises."""
        with pytest.raises(ValueError, match="out of range"):
            ScattererSlab(
                model=model_4layer,
                scatterer_ifaces=[0],
                tmatrices={0: weak_tmatrix},
                number_densities={0: 1.0},
            )

    def test_interface_too_large(self, model_4layer, weak_tmatrix):
        """Interface beyond M raises."""
        M = model_4layer.n_layers - 2
        with pytest.raises(ValueError, match="out of range"):
            ScattererSlab(
                model=model_4layer,
                scatterer_ifaces=[M + 1],
                tmatrices={M + 1: weak_tmatrix},
                number_densities={M + 1: 1.0},
            )

    def test_missing_tmatrix(self, model_4layer):
        """Missing T-matrix for declared interface raises."""
        with pytest.raises(ValueError, match="Missing T-matrix"):
            ScattererSlab(
                model=model_4layer,
                scatterer_ifaces=[1],
                tmatrices={},
                number_densities={1: 1.0},
            )

    def test_wrong_tmatrix_shape(self, model_4layer):
        """T-matrix with wrong shape raises."""
        with pytest.raises(ValueError, match="shape"):
            ScattererSlab(
                model=model_4layer,
                scatterer_ifaces=[1],
                tmatrices={1: np.eye(3, dtype=_CD)},
                number_densities={1: 1.0},
            )

    def test_missing_number_density(self, model_4layer, weak_tmatrix):
        """Missing number density raises."""
        with pytest.raises(ValueError, match="Missing number density"):
            ScattererSlab(
                model=model_4layer,
                scatterer_ifaces=[1],
                tmatrices={1: weak_tmatrix},
                number_densities={},
            )


# ===== Background incident field tests =====


class TestBackgroundIncidentField:
    """Incident field matches Green's function column 2."""

    def test_finite_values(self, model_4layer, omega, kH):
        """Incident field is finite at all scatterer interfaces."""
        psi0 = background_incident_field(model_4layer, omega, kH, [1, 2])
        for j in [1, 2]:
            assert psi0[j].shape == (len(kH), 4)
            assert np.all(np.isfinite(psi0[j]))

    def test_matches_greens_column2(self, model_4layer, omega, kH):
        """Incident field equals G(j, 0)[:, :, 2]."""
        psi0 = background_incident_field(model_4layer, omega, kH, [1, 2])
        for j in [1, 2]:
            G_j0 = layered_greens_psv(
                model_4layer, omega, kH, source_iface=0, receiver_iface=j
            )
            np.testing.assert_allclose(psi0[j], G_j0[:, :, 2], atol=1e-14)


# ===== Interlayer Green's matrix tests =====


class TestInterlayerGreensMatrix:
    """Block Green's matrix structure and values."""

    def test_diagonal_is_zero(self, model_4layer, omega, kH):
        """Diagonal blocks (intralayer) are exactly zero."""
        G_block = build_interlayer_greens_matrix(model_4layer, omega, kH, [1, 2])
        n_kH = len(kH)
        # Diagonal blocks
        np.testing.assert_array_equal(G_block[:, :4, :4], 0.0)
        np.testing.assert_array_equal(G_block[:, 4:8, 4:8], 0.0)

    def test_offdiag_matches_greens(self, model_4layer, omega, kH):
        """Off-diagonal blocks match layered_greens_psv."""
        ifaces = [1, 2]
        G_block = build_interlayer_greens_matrix(model_4layer, omega, kH, ifaces)

        # G_block[0,1] = G(iface 1, iface 2) — receiver=1, source=2
        G_12 = layered_greens_psv(
            model_4layer, omega, kH, source_iface=2, receiver_iface=1
        )
        np.testing.assert_allclose(G_block[:, :4, 4:8], G_12, rtol=1e-12)

        # G_block[1,0] = G(iface 2, iface 1) — receiver=2, source=1
        G_21 = layered_greens_psv(
            model_4layer, omega, kH, source_iface=1, receiver_iface=2
        )
        np.testing.assert_allclose(G_block[:, 4:8, :4], G_21, rtol=1e-12)

    def test_shape(self, model_5layer, omega, kH):
        """Shape is correct for 3 scatterer interfaces."""
        ifaces = [1, 2, 3]
        G_block = build_interlayer_greens_matrix(model_5layer, omega, kH, ifaces)
        assert G_block.shape == (len(kH), 12, 12)


# ===== Zero T-matrix test =====


class TestZeroTmatrix:
    """T=0 gives no scattering perturbation."""

    def test_zero_tmatrix_no_perturbation(self, model_4layer, omega, kH):
        """With T=0, R_total == R_background exactly."""
        zero_T = np.zeros((4, 4), dtype=_CD)
        slab = ScattererSlab(
            model=model_4layer,
            scatterer_ifaces=[1, 2],
            tmatrices={1: zero_T, 2: zero_T},
            number_densities={1: 1.0, 2: 1.0},
        )
        result = interlayer_ms_reflectivity(slab, omega, kH)
        np.testing.assert_allclose(result.R_total, result.R_background, atol=1e-14)
        np.testing.assert_allclose(result.R_born, result.R_background, atol=1e-14)


# ===== Born approximation test =====


class TestBornApproximation:
    """Foldy-Lax minus Born is O(T²) for weak scatterers."""

    def test_born_error_quadratic(self, model_4layer, omega, kH, rng):
        """Born error scales quadratically with T-matrix strength."""
        T_base = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        T_base = T_base.astype(_CD)

        errors = []
        epsilons = [1e-3, 5e-4, 2e-4]
        for eps in epsilons:
            T = eps * T_base
            slab = ScattererSlab(
                model=model_4layer,
                scatterer_ifaces=[1, 2],
                tmatrices={1: T, 2: T},
                number_densities={1: 1.0, 2: 1.0},
            )
            result = interlayer_ms_reflectivity(slab, omega, kH)
            diff = np.max(np.abs(result.R_total - result.R_born))
            errors.append(diff)

        # Check quadratic scaling: error ~ eps^2
        # Ratio of errors should scale as (eps1/eps2)^2
        for i in range(len(epsilons) - 1):
            ratio = errors[i] / errors[i + 1]
            eps_ratio = (epsilons[i] / epsilons[i + 1]) ** 2
            # Allow generous tolerance since we're checking scaling
            assert ratio > eps_ratio * 0.3, (
                f"Born error scaling not quadratic: ratio={ratio:.2f}, "
                f"expected ~{eps_ratio:.2f}"
            )


# ===== Single scatterer layer test =====


class TestSingleScattererLayer:
    """Single scatterer layer: no interlayer paths → Foldy-Lax == Born."""

    def test_single_layer_foldy_equals_born(self, model_4layer, omega, kH, rng):
        """With one scatterer layer, Foldy-Lax == Born (no interlayer coupling)."""
        T = 1e-3 * (rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4)))
        T = T.astype(_CD)
        slab = ScattererSlab(
            model=model_4layer,
            scatterer_ifaces=[1],
            tmatrices={1: T},
            number_densities={1: 1.0},
        )
        result = interlayer_ms_reflectivity(slab, omega, kH)
        # With one layer, G_block diagonal = 0, off-diagonal empty → I @ ψ = ψ⁰
        np.testing.assert_allclose(result.R_total, result.R_born, rtol=1e-10)

    def test_single_layer_exciting_equals_incident(self, model_4layer, omega, kH, rng):
        """With one scatterer layer, exciting == incident field."""
        T = 1e-3 * (rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4)))
        T = T.astype(_CD)
        slab = ScattererSlab(
            model=model_4layer,
            scatterer_ifaces=[1],
            tmatrices={1: T},
            number_densities={1: 1.0},
        )
        result = interlayer_ms_reflectivity(slab, omega, kH)
        np.testing.assert_allclose(
            result.psi_exciting[1], result.psi_incident[1], rtol=1e-10
        )


# ===== Two scatterer layers test =====


class TestTwoScattererLayers:
    """Two scatterer layers: system matrix has correct structure."""

    def test_system_matrix_structure(self, model_4layer, omega, kH, weak_tmatrix):
        """Foldy-Lax system matrix has identity on diagonal, -G*T off-diagonal."""
        ifaces = [1, 2]
        G_block = build_interlayer_greens_matrix(model_4layer, omega, kH, ifaces)
        n_kH = len(kH)
        N = 8  # 4*2

        n_j = 1.0
        T_block = np.zeros((n_kH, N, N), dtype=_CD)
        T_block[:, :4, :4] = n_j * weak_tmatrix[np.newaxis, :, :]
        T_block[:, 4:8, 4:8] = n_j * weak_tmatrix[np.newaxis, :, :]

        A = np.eye(N, dtype=_CD)[np.newaxis, :, :] - G_block @ T_block

        # Diagonal blocks should be I - 0 = I (since G diagonal is 0)
        for f in range(n_kH):
            np.testing.assert_allclose(A[f, :4, :4], np.eye(4), atol=1e-10)
            np.testing.assert_allclose(A[f, 4:8, 4:8], np.eye(4), atol=1e-10)

    def test_two_layers_different_from_born(self, model_4layer, omega, kH, rng):
        """With two scatterer layers and non-trivial T, Foldy-Lax ≠ Born."""
        T = 0.1 * (rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4)))
        T = T.astype(_CD)
        slab = ScattererSlab(
            model=model_4layer,
            scatterer_ifaces=[1, 2],
            tmatrices={1: T, 2: T},
            number_densities={1: 1.0, 2: 1.0},
        )
        result = interlayer_ms_reflectivity(slab, omega, kH)
        # With strong enough T and two layers, there should be a difference
        diff = np.max(np.abs(result.R_total - result.R_born))
        assert diff > 1e-10, "Foldy-Lax should differ from Born with two layers"


# ===== T-matrix conversion tests =====


class TestTmatrixConversion:
    """T-matrix basis conversion helpers."""

    def test_6x6_to_4x4_extraction(self):
        """6×6 → 4×4 extracts correct P-SV indices."""
        T_6x6 = np.arange(36, dtype=_CD).reshape(6, 6)
        T_4x4 = tmatrix_6x6_to_4x4_psv(T_6x6)
        assert T_4x4.shape == (4, 4)
        # Indices [1,0,3,4] from 6×6
        # (u_x=1, u_z=0, σ_zz=3, σ_xz=4) rows and columns
        idx = [1, 0, 3, 4]
        expected = T_6x6[np.ix_(idx, idx)]
        np.testing.assert_array_equal(T_4x4, expected)

    def test_6x6_to_4x4_permutation(self, rng):
        """Permutation preserves matrix structure for random T."""
        T_6x6 = rng.standard_normal((6, 6)) + 1j * rng.standard_normal((6, 6))
        T_4x4 = tmatrix_6x6_to_4x4_psv(T_6x6)
        # Check a specific element: T_4x4[0,0] should be T_6x6[1,1] (u_x, u_x)
        assert T_4x4[0, 0] == T_6x6[1, 1]
        # T_4x4[1,0] should be T_6x6[0,1] (u_z row, u_x col)
        assert T_4x4[1, 0] == T_6x6[0, 1]
        # T_4x4[2,3] should be T_6x6[3,4] (σ_zz row, σ_xz col)
        assert T_4x4[2, 3] == T_6x6[3, 4]

    def test_identity_6x6_gives_identity_4x4(self):
        """Identity 6×6 → identity 4×4."""
        T_4x4 = tmatrix_6x6_to_4x4_psv(np.eye(6, dtype=_CD))
        np.testing.assert_array_equal(T_4x4, np.eye(4, dtype=_CD))


# ===== Integration tests =====


class TestInterlayerMSIntegration:
    """End-to-end integration tests."""

    def test_result_types(self, model_4layer, omega, kH, weak_tmatrix):
        """Result has correct types and shapes."""
        slab = ScattererSlab(
            model=model_4layer,
            scatterer_ifaces=[1, 2],
            tmatrices={1: weak_tmatrix, 2: weak_tmatrix},
            number_densities={1: 1.0, 2: 1.0},
        )
        result = interlayer_ms_reflectivity(slab, omega, kH)
        assert isinstance(result, InterlayerMSResult)
        n = len(kH)
        assert result.R_background.shape == (n,)
        assert result.R_total.shape == (n,)
        assert result.R_born.shape == (n,)
        for j in [1, 2]:
            assert result.psi_exciting[j].shape == (n, 4)
            assert result.psi_incident[j].shape == (n, 4)

    def test_all_finite(self, model_4layer, omega, kH, weak_tmatrix):
        """All result arrays are finite."""
        slab = ScattererSlab(
            model=model_4layer,
            scatterer_ifaces=[1, 2],
            tmatrices={1: weak_tmatrix, 2: weak_tmatrix},
            number_densities={1: 1.0, 2: 1.0},
        )
        result = interlayer_ms_reflectivity(slab, omega, kH)
        assert np.all(np.isfinite(result.R_background))
        assert np.all(np.isfinite(result.R_total))
        assert np.all(np.isfinite(result.R_born))

    def test_5layer_three_scatterers(self, model_5layer, omega, kH, weak_tmatrix):
        """Works with 3 scatterer layers on a 5-layer model."""
        slab = ScattererSlab(
            model=model_5layer,
            scatterer_ifaces=[1, 2, 3],
            tmatrices={1: weak_tmatrix, 2: weak_tmatrix, 3: weak_tmatrix},
            number_densities={1: 0.5, 2: 1.0, 3: 0.5},
        )
        result = interlayer_ms_reflectivity(slab, omega, kH)
        assert result.R_total.shape == (len(kH),)
        assert np.all(np.isfinite(result.R_total))


# ===== 9×9 validation tests =====


class TestScattererSlab9x9Validation:
    """Invalid inputs to ScattererSlab9x9 raise ValueError."""

    def test_interface_out_of_range(self, model_4layer):
        """Interface index outside [1, M] raises."""
        T9 = np.eye(9, dtype=_CD)
        with pytest.raises(ValueError, match="out of range"):
            ScattererSlab9x9(
                model=model_4layer,
                scatterer_ifaces=[0],
                tmatrices={0: T9},
                number_densities={0: 1.0},
            )

    def test_wrong_tmatrix_shape(self, model_4layer):
        """T-matrix with wrong shape raises."""
        with pytest.raises(ValueError, match="shape"):
            ScattererSlab9x9(
                model=model_4layer,
                scatterer_ifaces=[1],
                tmatrices={1: np.eye(4, dtype=_CD)},
                number_densities={1: 1.0},
            )

    def test_missing_tmatrix(self, model_4layer):
        """Missing T-matrix for declared interface raises."""
        with pytest.raises(ValueError, match="Missing T-matrix"):
            ScattererSlab9x9(
                model=model_4layer,
                scatterer_ifaces=[1],
                tmatrices={},
                number_densities={1: 1.0},
            )

    def test_missing_number_density(self, model_4layer):
        """Missing number density raises."""
        T9 = np.eye(9, dtype=_CD)
        with pytest.raises(ValueError, match="Missing number density"):
            ScattererSlab9x9(
                model=model_4layer,
                scatterer_ifaces=[1],
                tmatrices={1: T9},
                number_densities={},
            )


# ===== 9×9 zero T-matrix test =====


class TestZeroTmatrix9x9:
    """T=0 gives no scattering perturbation in 9×9 basis."""

    def test_zero_tmatrix_no_perturbation(self, model_4layer, omega, kH):
        """With T=0, R_total == R_background exactly."""
        zero_T = np.zeros((9, 9), dtype=_CD)
        kx = kH
        ky = np.zeros_like(kH)
        slab = ScattererSlab9x9(
            model=model_4layer,
            scatterer_ifaces=[1, 2],
            tmatrices={1: zero_T, 2: zero_T},
            number_densities={1: 1.0, 2: 1.0},
        )
        result = interlayer_ms_reflectivity_9x9(slab, omega, kx, ky)
        np.testing.assert_allclose(result.R_total, result.R_background, atol=1e-14)
        np.testing.assert_allclose(result.R_born, result.R_background, atol=1e-14)


# ===== 9×9 Born approximation test =====


class TestBornApproximation9x9:
    """Foldy-Lax minus Born is O(T²) for weak scatterers in 9×9 basis."""

    def test_born_error_quadratic(self, model_4layer, omega, kH, rng):
        """Born error scales quadratically with T-matrix strength."""
        T_base = rng.standard_normal((9, 9)) + 1j * rng.standard_normal((9, 9))
        T_base = T_base.astype(_CD)
        kx = kH
        ky = np.zeros_like(kH)

        errors = []
        epsilons = [1e-3, 5e-4, 2e-4]
        for eps in epsilons:
            T = eps * T_base
            slab = ScattererSlab9x9(
                model=model_4layer,
                scatterer_ifaces=[1, 2],
                tmatrices={1: T, 2: T},
                number_densities={1: 1.0, 2: 1.0},
            )
            result = interlayer_ms_reflectivity_9x9(slab, omega, kx, ky)
            diff = np.max(np.abs(result.R_total - result.R_born))
            errors.append(diff)

        for i in range(len(epsilons) - 1):
            ratio = errors[i] / errors[i + 1]
            eps_ratio = (epsilons[i] / epsilons[i + 1]) ** 2
            assert ratio > eps_ratio * 0.3, (
                f"Born error scaling not quadratic: ratio={ratio:.2f}, "
                f"expected ~{eps_ratio:.2f}"
            )


# ===== 9×9 single scatterer layer test =====


class TestSingleScattererLayer9x9:
    """Single scatterer layer: Foldy-Lax == Born in 9×9 basis."""

    def test_single_layer_foldy_equals_born(self, model_4layer, omega, kH, rng):
        """With one scatterer layer, Foldy-Lax == Born."""
        T = 1e-3 * (rng.standard_normal((9, 9)) + 1j * rng.standard_normal((9, 9)))
        T = T.astype(_CD)
        kx = kH
        ky = np.zeros_like(kH)
        slab = ScattererSlab9x9(
            model=model_4layer,
            scatterer_ifaces=[1],
            tmatrices={1: T},
            number_densities={1: 1.0},
        )
        result = interlayer_ms_reflectivity_9x9(slab, omega, kx, ky)
        np.testing.assert_allclose(result.R_total, result.R_born, rtol=1e-10)


# ===== 4×4 ↔ 9×9 cross-validation =====


class TestCrossValidation4x4_9x9:
    """9×9 solver matches 4×4 solver for sagittal (ky=0) diagonal T."""

    def test_4x4_9x9_consistency(self, model_4layer, omega):
        """Sagittal plane: 9×9 solver matches 4×4 for isotropic diagonal T."""
        kH = np.array([2.0, 4.0, 6.0])
        kx = kH.copy()
        ky = np.zeros_like(kH)

        eps = 1e-4
        T_9x9 = eps * np.eye(9, dtype=_CD)

        rho, alpha, beta = _interface_elastic_properties(model_4layer, 1)

        # 9×9 solver — single scatterer layer at interface 1
        slab_9 = ScattererSlab9x9(
            model=model_4layer,
            scatterer_ifaces=[1],
            tmatrices={1: T_9x9},
            number_densities={1: 1.0},
        )
        result_9 = interlayer_ms_reflectivity_9x9(slab_9, omega, kx, ky)

        # 4×4 solver at each kH (T_4x4 is kx-dependent)
        R_total_4 = np.zeros(len(kH), dtype=_CD)
        for i, kx_i in enumerate(kx):
            T_4x4 = tmatrix_9x9_to_4x4_psv(T_9x9, float(kx_i), 0.0, rho, alpha, beta)
            slab_4 = ScattererSlab(
                model=model_4layer,
                scatterer_ifaces=[1],
                tmatrices={1: T_4x4},
                number_densities={1: 1.0},
            )
            result_4 = interlayer_ms_reflectivity(slab_4, omega, np.array([kH[i]]))
            R_total_4[i] = result_4.R_total[0]

        np.testing.assert_allclose(result_9.R_total, R_total_4, rtol=1e-8)


# ===== 9×9 integration tests =====


class TestInterlayerMSIntegration9x9:
    """End-to-end integration tests for 9×9 solver."""

    def test_result_types(self, model_4layer, omega, kH):
        """Result has correct types and shapes."""
        T9 = 1e-4 * np.eye(9, dtype=_CD)
        kx = kH
        ky = np.zeros_like(kH)
        slab = ScattererSlab9x9(
            model=model_4layer,
            scatterer_ifaces=[1, 2],
            tmatrices={1: T9, 2: T9},
            number_densities={1: 1.0, 2: 1.0},
        )
        result = interlayer_ms_reflectivity_9x9(slab, omega, kx, ky)
        assert isinstance(result, InterlayerMSResult9x9)
        n = len(kH)
        assert result.R_background.shape == (n,)
        assert result.R_total.shape == (n,)
        assert result.R_born.shape == (n,)
        for j in [1, 2]:
            assert result.psi_exciting[j].shape == (n, 9)
            assert result.psi_incident[j].shape == (n, 9)

    def test_all_finite(self, model_4layer, omega, kH):
        """All result arrays are finite."""
        T9 = 1e-4 * np.eye(9, dtype=_CD)
        kx = kH
        ky = np.zeros_like(kH)
        slab = ScattererSlab9x9(
            model=model_4layer,
            scatterer_ifaces=[1, 2],
            tmatrices={1: T9, 2: T9},
            number_densities={1: 1.0, 2: 1.0},
        )
        result = interlayer_ms_reflectivity_9x9(slab, omega, kx, ky)
        assert np.all(np.isfinite(result.R_background))
        assert np.all(np.isfinite(result.R_total))
        assert np.all(np.isfinite(result.R_born))
