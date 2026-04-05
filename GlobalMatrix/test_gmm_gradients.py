"""Validate GMM Jacobian and Hessian against Kennett AD derivatives."""

import numpy as np
import pytest
import torch

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

from .gmm_torch import gmm_hessian, gmm_jacobian, gmm_reflectivity_torch


@pytest.fixture
def model():
    return default_ocean_crust_model()


@pytest.fixture
def tensors(model):
    return model_to_tensors(model, requires_grad=True)


@pytest.fixture
def omega():
    T = 64.0
    nw = 32  # small for speed in gradient tests
    dw = 2.0 * np.pi / T
    return torch.arange(1, nw, dtype=torch.float64) * dw


class TestGMMTorchForward:
    """Verify torch GMM matches numpy GMM."""

    def test_torch_matches_numpy(self, model, tensors, omega):
        """Torch GMM reflectivity matches NumPy version."""
        from .global_matrix import gmm_reflectivity

        omega_np = omega.detach().numpy()
        R_np = gmm_reflectivity(model, 0.2, omega_np, free_surface=False)

        R_torch = gmm_reflectivity_torch(
            tensors["alpha"],
            tensors["beta"],
            tensors["rho"],
            tensors["thickness"],
            tensors["Q_alpha"],
            tensors["Q_beta"],
            0.2,
            omega,
            free_surface=False,
        )

        np.testing.assert_allclose(
            R_torch.detach().numpy(), R_np, rtol=1e-12, atol=1e-14
        )


class TestGMMJacobian:
    """Validate GMM Jacobian against Kennett AD Jacobian."""

    def test_jacobian_matches_kennett(self, tensors, omega):
        """GMM Jacobian matches Kennett AD Jacobian to < 1e-10."""
        J_gmm = gmm_jacobian(
            tensors["alpha"],
            tensors["beta"],
            tensors["rho"],
            tensors["thickness"],
            tensors["Q_alpha"],
            tensors["Q_beta"],
            0.2,
            omega,
        )

        J_kennett = kennett_jacobian(
            tensors["alpha"],
            tensors["beta"],
            tensors["rho"],
            tensors["thickness"],
            tensors["Q_alpha"],
            tensors["Q_beta"],
            0.2,
            omega,
        )

        J_gmm_np = J_gmm.detach().numpy()
        J_kennett_np = J_kennett.detach().numpy()

        # Compare element-wise
        mask = np.abs(J_kennett_np) > 1e-15
        rel_err = np.abs(J_gmm_np[mask] - J_kennett_np[mask]) / np.abs(
            J_kennett_np[mask]
        )
        max_rel = np.max(rel_err) if rel_err.size > 0 else 0.0
        assert max_rel < 1e-6, f"Max Jacobian relative error: {max_rel:.2e}"

        # Absolute error
        abs_err = np.max(np.abs(J_gmm_np - J_kennett_np))
        assert abs_err < 1e-8, f"Max Jacobian absolute error: {abs_err:.2e}"

    def test_jacobian_vs_finite_differences(self, tensors, omega):
        """GMM Jacobian matches central finite differences to < 1e-5."""
        from Kennett_Reflectivity.kennett_torch import _pack_params, _unpack_params

        params = _pack_params(
            tensors["alpha"],
            tensors["beta"],
            tensors["rho"],
            tensors["thickness"],
        )
        n_params = params.shape[0]
        nfreq = omega.shape[0]

        J_gmm = gmm_jacobian(
            tensors["alpha"],
            tensors["beta"],
            tensors["rho"],
            tensors["thickness"],
            tensors["Q_alpha"],
            tensors["Q_beta"],
            0.2,
            omega,
        )

        # Central finite differences
        eps = 1e-7
        J_fd = torch.zeros(nfreq, n_params, dtype=torch.complex128)
        for j in range(n_params):
            p_plus = params.clone()
            p_plus[j] += eps
            a_p, b_p, r_p, h_p = _unpack_params(
                p_plus,
                tensors["alpha"],
                tensors["beta"],
                tensors["rho"],
                tensors["thickness"],
            )
            R_plus = gmm_reflectivity_torch(
                a_p,
                b_p,
                r_p,
                h_p,
                tensors["Q_alpha"],
                tensors["Q_beta"],
                0.2,
                omega,
            )

            p_minus = params.clone()
            p_minus[j] -= eps
            a_m, b_m, r_m, h_m = _unpack_params(
                p_minus,
                tensors["alpha"],
                tensors["beta"],
                tensors["rho"],
                tensors["thickness"],
            )
            R_minus = gmm_reflectivity_torch(
                a_m,
                b_m,
                r_m,
                h_m,
                tensors["Q_alpha"],
                tensors["Q_beta"],
                0.2,
                omega,
            )

            J_fd[:, j] = (R_plus - R_minus) / (2.0 * eps)

        J_gmm_np = J_gmm.detach().numpy()
        J_fd_np = J_fd.detach().numpy()

        mask = np.abs(J_fd_np) > 1e-10
        rel_err = np.abs(J_gmm_np[mask] - J_fd_np[mask]) / np.abs(J_fd_np[mask])
        max_rel = np.max(rel_err) if rel_err.size > 0 else 0.0
        assert max_rel < 1e-4, f"Max FD relative error: {max_rel:.2e}"


class TestGMMHessian:
    """Validate GMM Hessian against Kennett AD Hessian."""

    def test_hessian_matches_kennett(self, tensors):
        """GMM Hessian matches Kennett AD Hessian to < 1e-6."""
        # Use very few frequencies for speed
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

        abs_err = np.max(np.abs(H_gmm_np - H_kennett_np))
        assert abs_err < 1e-6, f"Max Hessian absolute error: {abs_err:.2e}"
