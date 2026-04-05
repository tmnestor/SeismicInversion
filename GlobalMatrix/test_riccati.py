"""Tests for the Block-Riccati sweep solver."""

import numpy as np
import pytest
import torch

from Kennett_Reflectivity.kennett_reflectivity import kennett_reflectivity
from Kennett_Reflectivity.kennett_seismogram import default_ocean_crust_model
from Kennett_Reflectivity.kennett_torch import (
    hessian as kennett_hessian,
)
from Kennett_Reflectivity.kennett_torch import (
    jacobian as kennett_jacobian,
)
from Kennett_Reflectivity.kennett_torch import (
    model_to_tensors,
)
from Kennett_Reflectivity.layer_model import LayerModel

from .global_matrix import gmm_reflectivity
from .gmm_torch import gmm_hessian, gmm_jacobian, gmm_reflectivity_torch


@pytest.fixture
def model():
    return default_ocean_crust_model()


@pytest.fixture
def omega():
    T = 64.0
    nw = 256
    dw = 2.0 * np.pi / T
    return np.arange(1, nw, dtype=np.float64) * dw


@pytest.fixture
def omega_torch():
    T = 64.0
    nw = 32
    dw = 2.0 * np.pi / T
    return torch.arange(1, nw, dtype=torch.float64) * dw


@pytest.fixture
def tensors(model):
    return model_to_tensors(model, requires_grad=True)


# ===== Riccati vs Dense solve agreement =====


class TestRiccatiVsDense:
    """Verify Riccati and dense solvers produce identical results."""

    @pytest.mark.parametrize("p", [0.1, 0.2, 0.3, 0.4, 0.6])
    def test_riccati_matches_dense(self, model, omega, p):
        """Riccati matches dense solve to < 1e-12 relative error."""
        R_riccati = gmm_reflectivity(model, p, omega, solver="riccati")
        R_dense = gmm_reflectivity(model, p, omega, solver="dense")

        mask = np.abs(R_dense) > 1e-15
        rel_err = np.abs(R_riccati[mask] - R_dense[mask]) / np.abs(R_dense[mask])
        assert np.max(rel_err) < 1e-10, f"Max relative error: {np.max(rel_err):.2e}"

        abs_err = np.abs(R_riccati - R_dense)
        assert np.max(abs_err) < 1e-12, f"Max absolute error: {np.max(abs_err):.2e}"

    @pytest.mark.parametrize("p", [0.1, 0.3, 0.6])
    def test_free_surface_matches_dense(self, model, omega, p):
        """Free-surface Riccati matches dense solve."""
        R_riccati = gmm_reflectivity(
            model, p, omega, free_surface=True, solver="riccati"
        )
        R_dense = gmm_reflectivity(model, p, omega, free_surface=True, solver="dense")

        mask = np.abs(R_dense) > 1e-15
        rel_err = np.abs(R_riccati[mask] - R_dense[mask]) / np.abs(R_dense[mask])
        assert np.max(rel_err) < 1e-10, f"Max relative error: {np.max(rel_err):.2e}"


# ===== Riccati vs Kennett =====


class TestRiccatiVsKennett:
    """Verify Riccati solver matches Kennett reflectivity."""

    @pytest.mark.parametrize("p", [0.1, 0.2, 0.3, 0.4, 0.6])
    def test_riccati_matches_kennett(self, model, omega, p):
        """Riccati R(w,p) matches Kennett to < 1e-12."""
        R_kennett = kennett_reflectivity(model, p, omega, free_surface=False)
        R_riccati = gmm_reflectivity(model, p, omega, solver="riccati")

        mask = np.abs(R_kennett) > 1e-15
        rel_err = np.abs(R_riccati[mask] - R_kennett[mask]) / np.abs(R_kennett[mask])
        assert np.max(rel_err) < 1e-10, f"Max relative error: {np.max(rel_err):.2e}"

        abs_err = np.abs(R_riccati - R_kennett)
        assert np.max(abs_err) < 1e-12, f"Max absolute error: {np.max(abs_err):.2e}"


# ===== PyTorch Riccati vs Dense =====


class TestRiccatiTorch:
    """Verify PyTorch Riccati matches dense and produces correct gradients."""

    def test_torch_riccati_matches_dense(self, tensors, omega_torch):
        """PyTorch Riccati matches dense solver."""
        R_riccati = gmm_reflectivity_torch(
            tensors["alpha"],
            tensors["beta"],
            tensors["rho"],
            tensors["thickness"],
            tensors["Q_alpha"],
            tensors["Q_beta"],
            0.2,
            omega_torch,
            solver="riccati",
        )
        R_dense = gmm_reflectivity_torch(
            tensors["alpha"],
            tensors["beta"],
            tensors["rho"],
            tensors["thickness"],
            tensors["Q_alpha"],
            tensors["Q_beta"],
            0.2,
            omega_torch,
            solver="dense",
        )

        np.testing.assert_allclose(
            R_riccati.detach().numpy(), R_dense.detach().numpy(), rtol=1e-10, atol=1e-12
        )

    def test_jacobian_riccati_matches_kennett(self, tensors, omega_torch):
        """Riccati Jacobian matches Kennett AD Jacobian."""
        J_gmm = gmm_jacobian(
            tensors["alpha"],
            tensors["beta"],
            tensors["rho"],
            tensors["thickness"],
            tensors["Q_alpha"],
            tensors["Q_beta"],
            0.2,
            omega_torch,
        )
        J_kennett = kennett_jacobian(
            tensors["alpha"],
            tensors["beta"],
            tensors["rho"],
            tensors["thickness"],
            tensors["Q_alpha"],
            tensors["Q_beta"],
            0.2,
            omega_torch,
        )

        J_gmm_np = J_gmm.detach().numpy()
        J_kennett_np = J_kennett.detach().numpy()

        mask = np.abs(J_kennett_np) > 1e-15
        rel_err = np.abs(J_gmm_np[mask] - J_kennett_np[mask]) / np.abs(
            J_kennett_np[mask]
        )
        max_rel = np.max(rel_err) if rel_err.size > 0 else 0.0
        assert max_rel < 1e-6, f"Max Jacobian relative error: {max_rel:.2e}"

    def test_hessian_riccati_matches_kennett(self, tensors):
        """Riccati Hessian matches Kennett AD Hessian."""
        T = 64.0
        nw = 8
        dw = 2.0 * np.pi / T
        omega = torch.arange(1, nw, dtype=torch.float64) * dw

        H_gmm = gmm_hessian(
            tensors["alpha"],
            tensors["beta"],
            tensors["rho"],
            tensors["thickness"],
            tensors["Q_alpha"],
            tensors["Q_beta"],
            0.2,
            omega,
        )
        H_kennett = kennett_hessian(
            tensors["alpha"],
            tensors["beta"],
            tensors["rho"],
            tensors["thickness"],
            tensors["Q_alpha"],
            tensors["Q_beta"],
            0.2,
            omega,
        )

        H_gmm_np = H_gmm.detach().numpy()
        H_kennett_np = H_kennett.detach().numpy()

        mask = np.abs(H_kennett_np) > 1e-10
        rel_err = np.abs(H_gmm_np[mask] - H_kennett_np[mask]) / np.abs(
            H_kennett_np[mask]
        )
        max_rel = np.max(rel_err) if rel_err.size > 0 else 0.0
        assert max_rel < 1e-4, f"Max Hessian relative error: {max_rel:.2e}"


# ===== Edge cases =====


class TestEdgeCases:
    """Edge cases: 2-layer, 3-layer, evanescent modes."""

    def test_two_layer_model(self, omega):
        """2-layer model (ocean + half-space): M=0, no Riccati steps."""
        model = LayerModel.from_arrays(
            alpha=[1.5, 5.0],
            beta=[0.0, 3.0],
            rho=[1.0, 3.0],
            thickness=[2.0, np.inf],
            Q_alpha=[20000, 100],
            Q_beta=[1e10, 100],
        )
        R_riccati = gmm_reflectivity(model, 0.2, omega, solver="riccati")
        R_dense = gmm_reflectivity(model, 0.2, omega, solver="dense")
        R_kennett = kennett_reflectivity(model, 0.2, omega, free_surface=False)

        np.testing.assert_allclose(R_riccati, R_dense, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(R_riccati, R_kennett, rtol=1e-10, atol=1e-12)

    def test_three_layer_model(self, omega):
        """3-layer model (ocean + 1 elastic + half-space): single Riccati step."""
        model = LayerModel.from_arrays(
            alpha=[1.5, 3.0, 5.0],
            beta=[0.0, 1.5, 3.0],
            rho=[1.0, 3.0, 3.0],
            thickness=[2.0, 1.0, np.inf],
            Q_alpha=[20000, 100, 100],
            Q_beta=[1e10, 100, 100],
        )
        R_riccati = gmm_reflectivity(model, 0.2, omega, solver="riccati")
        R_dense = gmm_reflectivity(model, 0.2, omega, solver="dense")
        R_kennett = kennett_reflectivity(model, 0.2, omega, free_surface=False)

        np.testing.assert_allclose(R_riccati, R_dense, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(R_riccati, R_kennett, rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize("p", [0.4, 0.6, 0.8])
    def test_evanescent_modes(self, model, omega, p):
        """Evanescent modes at large p: all values finite."""
        R = gmm_reflectivity(model, p, omega, solver="riccati")
        assert np.all(np.isfinite(R)), "Non-finite values in evanescent regime"


# ===== Many-layer model =====


class TestManyLayers:
    """Verify correctness with 20+ layers."""

    @staticmethod
    def _make_many_layer_model(n_elastic: int = 20) -> LayerModel:
        """Create a model with n_elastic finite elastic layers + ocean + half-space."""
        rng = np.random.default_rng(42)
        n_total = n_elastic + 2  # ocean + elastic layers + half-space

        alpha = np.zeros(n_total)
        beta = np.zeros(n_total)
        rho = np.zeros(n_total)
        thickness = np.zeros(n_total)
        Q_alpha = np.zeros(n_total)
        Q_beta = np.zeros(n_total)

        # Ocean
        alpha[0] = 1.5
        beta[0] = 0.0
        rho[0] = 1.0
        thickness[0] = 2.0
        Q_alpha[0] = 20000
        Q_beta[0] = 1e10

        # Elastic layers with gradually increasing velocity
        for i in range(1, n_elastic + 1):
            alpha[i] = 2.0 + 0.15 * i + rng.uniform(-0.05, 0.05)
            beta[i] = alpha[i] * 0.5 + rng.uniform(-0.02, 0.02)
            rho[i] = 2.0 + 0.05 * i + rng.uniform(-0.02, 0.02)
            thickness[i] = 0.5 + rng.uniform(0, 0.5)
            Q_alpha[i] = 100
            Q_beta[i] = 100

        # Half-space
        alpha[-1] = alpha[-2] + 0.5
        beta[-1] = beta[-2] + 0.3
        rho[-1] = rho[-2] + 0.2
        thickness[-1] = np.inf
        Q_alpha[-1] = 100
        Q_beta[-1] = 100

        return LayerModel.from_arrays(
            alpha=alpha.tolist(),
            beta=beta.tolist(),
            rho=rho.tolist(),
            thickness=thickness.tolist(),
            Q_alpha=Q_alpha.tolist(),
            Q_beta=Q_beta.tolist(),
        )

    def test_many_layer_correctness(self):
        """20-layer model: Riccati matches dense solve."""
        model = self._make_many_layer_model(20)
        T = 64.0
        nw = 64
        dw = 2.0 * np.pi / T
        omega = np.arange(1, nw, dtype=np.float64) * dw

        R_riccati = gmm_reflectivity(model, 0.15, omega, solver="riccati")
        R_dense = gmm_reflectivity(model, 0.15, omega, solver="dense")

        mask = np.abs(R_dense) > 1e-15
        rel_err = np.abs(R_riccati[mask] - R_dense[mask]) / np.abs(R_dense[mask])
        assert np.max(rel_err) < 1e-8, f"Max relative error: {np.max(rel_err):.2e}"

    def test_many_layer_matches_kennett(self):
        """20-layer model: Riccati matches Kennett."""
        model = self._make_many_layer_model(20)
        T = 64.0
        nw = 64
        dw = 2.0 * np.pi / T
        omega = np.arange(1, nw, dtype=np.float64) * dw

        R_riccati = gmm_reflectivity(model, 0.15, omega, solver="riccati")
        R_kennett = kennett_reflectivity(model, 0.15, omega, free_surface=False)

        mask = np.abs(R_kennett) > 1e-15
        rel_err = np.abs(R_riccati[mask] - R_kennett[mask]) / np.abs(R_kennett[mask])
        assert np.max(rel_err) < 1e-8, f"Max relative error: {np.max(rel_err):.2e}"

    def test_many_layer_finite(self):
        """20-layer model: all values finite."""
        model = self._make_many_layer_model(20)
        T = 64.0
        nw = 64
        dw = 2.0 * np.pi / T
        omega = np.arange(1, nw, dtype=np.float64) * dw

        R = gmm_reflectivity(model, 0.3, omega, solver="riccati")
        assert np.all(np.isfinite(R)), "Non-finite values in many-layer model"
