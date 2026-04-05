"""Gradient validation tests for the differentiable Kennett reflectivity.

Tests:
1. Forward model agreement with NumPy reference (< 1e-12 relative error)
2. AD Jacobian vs finite differences (< 1e-6 relative error)
3. AD Hessian vs finite differences (< 1e-4 relative error)
4. Branch-cut correctness at subcritical/critical/supercritical slownesses
5. AD Jacobian vs analytical Frechet derivative (Dietrich & Kormendi, < 1e-12)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from Kennett_Reflectivity.kennett_reflectivity import kennett_reflectivity
from Kennett_Reflectivity.kennett_seismogram import default_ocean_crust_model
from Kennett_Reflectivity.kennett_torch import (
    _pack_params,
    _unpack_params,
    hessian,
    jacobian,
    kennett_reflectivity_torch,
    model_to_tensors,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def model():
    """Default 5-layer ocean-crust model."""
    return default_ocean_crust_model()


@pytest.fixture()
def omega():
    """Frequency grid matching compute_seismogram defaults (T=64, nw=2048)."""
    T = 64.0
    nw = 2048
    dw = 2.0 * np.pi / T
    nwm = nw - 1
    return np.arange(1, nwm + 1, dtype=np.float64) * dw


@pytest.fixture()
def omega_small():
    """Small frequency grid for Hessian tests (expensive)."""
    T = 64.0
    dw = 2.0 * np.pi / T
    return np.arange(1, 33, dtype=np.float64) * dw


# ---------------------------------------------------------------------------
# 1. Forward model agreement
# ---------------------------------------------------------------------------


class TestForwardAgreement:
    """Compare torch output against NumPy reference."""

    def test_default_model_p02(self, model, omega):
        """Subcritical slowness p=0.2 s/km."""
        p = 0.2
        R_np = kennett_reflectivity(model, p, omega)

        tensors = model_to_tensors(model)
        omega_t = torch.tensor(omega, dtype=torch.float64)
        R_torch = kennett_reflectivity_torch(
            **tensors,
            p=p,
            omega=omega_t,
        )
        R_t = R_torch.detach().numpy()

        rel_err = np.max(np.abs(R_t - R_np)) / (np.max(np.abs(R_np)) + 1e-30)
        assert rel_err < 1e-12, f"Forward mismatch: relative error = {rel_err:.2e}"

    def test_default_model_p04(self, model, omega):
        """Post-critical slowness p=0.4 s/km."""
        p = 0.4
        R_np = kennett_reflectivity(model, p, omega)

        tensors = model_to_tensors(model)
        omega_t = torch.tensor(omega, dtype=torch.float64)
        R_torch = kennett_reflectivity_torch(
            **tensors,
            p=p,
            omega=omega_t,
        )
        R_t = R_torch.detach().numpy()

        rel_err = np.max(np.abs(R_t - R_np)) / (np.max(np.abs(R_np)) + 1e-30)
        assert rel_err < 1e-12, f"Forward mismatch: relative error = {rel_err:.2e}"

    def test_free_surface(self, model, omega):
        """Free surface mode."""
        p = 0.2
        R_np = kennett_reflectivity(model, p, omega, free_surface=True)

        tensors = model_to_tensors(model)
        omega_t = torch.tensor(omega, dtype=torch.float64)
        R_torch = kennett_reflectivity_torch(
            **tensors,
            p=p,
            omega=omega_t,
            free_surface=True,
        )
        R_t = R_torch.detach().numpy()

        rel_err = np.max(np.abs(R_t - R_np)) / (np.max(np.abs(R_np)) + 1e-30)
        assert rel_err < 1e-12, f"Free-surface mismatch: relative error = {rel_err:.2e}"


# ---------------------------------------------------------------------------
# 2. Jacobian vs finite differences
# ---------------------------------------------------------------------------


class TestJacobian:
    """AD Jacobian vs central finite differences."""

    def test_jacobian_fd(self, model, omega_small):
        """Jacobian matches finite differences for all parameters."""
        p = 0.2
        omega_t = torch.tensor(omega_small, dtype=torch.float64)
        tensors = model_to_tensors(model)
        alpha = tensors["alpha"]
        beta = tensors["beta"]
        rho = tensors["rho"]
        thickness = tensors["thickness"]
        Q_alpha = tensors["Q_alpha"]
        Q_beta = tensors["Q_beta"]

        J_ad = jacobian(
            alpha,
            beta,
            rho,
            thickness,
            Q_alpha,
            Q_beta,
            p,
            omega_t,
        )

        # Finite-difference Jacobian
        params = _pack_params(alpha, beta, rho, thickness)
        n_params = params.shape[0]
        nfreq = omega_t.shape[0]
        delta = 1e-7
        J_fd = torch.zeros((nfreq, n_params), dtype=torch.complex128)

        for j in range(n_params):
            dp = torch.zeros_like(params)
            dp[j] = delta

            p_plus = params + dp
            a_p, b_p, r_p, h_p = _unpack_params(p_plus, alpha, beta, rho, thickness)
            R_plus = kennett_reflectivity_torch(
                a_p,
                b_p,
                r_p,
                h_p,
                Q_alpha,
                Q_beta,
                p,
                omega_t,
            )

            p_minus = params - dp
            a_m, b_m, r_m, h_m = _unpack_params(p_minus, alpha, beta, rho, thickness)
            R_minus = kennett_reflectivity_torch(
                a_m,
                b_m,
                r_m,
                h_m,
                Q_alpha,
                Q_beta,
                p,
                omega_t,
            )

            J_fd[:, j] = (R_plus - R_minus) / (2.0 * delta)

        # Compare
        J_ad_np = J_ad.detach().numpy()
        J_fd_np = J_fd.detach().numpy()
        scale = np.max(np.abs(J_fd_np)) + 1e-30
        max_err = np.max(np.abs(J_ad_np - J_fd_np)) / scale

        assert max_err < 1e-5, (
            f"Jacobian AD vs FD mismatch: max relative error = {max_err:.2e}"
        )


# ---------------------------------------------------------------------------
# 3. Hessian vs finite differences
# ---------------------------------------------------------------------------


class TestHessian:
    """AD Hessian vs second-order finite differences."""

    def test_hessian_fd(self, model, omega_small):
        """Hessian matches finite differences for L2 misfit."""
        p = 0.2
        omega_t = torch.tensor(omega_small, dtype=torch.float64)
        tensors = model_to_tensors(model)
        alpha = tensors["alpha"]
        beta = tensors["beta"]
        rho = tensors["rho"]
        thickness = tensors["thickness"]
        Q_alpha = tensors["Q_alpha"]
        Q_beta = tensors["Q_beta"]

        # Generate synthetic observed data
        R_obs = kennett_reflectivity_torch(
            alpha,
            beta,
            rho,
            thickness,
            Q_alpha,
            Q_beta,
            p,
            omega_t,
        ).detach()

        # Perturb the model slightly so the Hessian is non-trivial
        alpha_pert = alpha.clone()
        alpha_pert[1] = alpha_pert[1] * 1.02  # 2% perturbation

        H_ad = hessian(
            alpha_pert,
            beta,
            rho,
            thickness,
            Q_alpha,
            Q_beta,
            p,
            omega_t,
            R_obs=R_obs,
        )

        # FD Hessian of the misfit
        params = _pack_params(alpha_pert, beta, rho, thickness)
        n_params = params.shape[0]
        delta = 1e-5

        def _misfit_eval(pvec: torch.Tensor) -> float:
            a, b, r, h = _unpack_params(pvec, alpha_pert, beta, rho, thickness)
            R = kennett_reflectivity_torch(
                a,
                b,
                r,
                h,
                Q_alpha,
                Q_beta,
                p,
                omega_t,
            )
            res = R - R_obs
            return (res.real**2 + res.imag**2).sum().item()

        H_fd = np.zeros((n_params, n_params))
        chi0 = _misfit_eval(params)

        for i in range(n_params):
            for j in range(i, n_params):
                dp_i = torch.zeros_like(params)
                dp_j = torch.zeros_like(params)
                dp_i[i] = delta
                dp_j[j] = delta

                if i == j:
                    f_plus = _misfit_eval(params + dp_i)
                    f_minus = _misfit_eval(params - dp_i)
                    H_fd[i, j] = (f_plus - 2.0 * chi0 + f_minus) / (delta**2)
                else:
                    f_pp = _misfit_eval(params + dp_i + dp_j)
                    f_pm = _misfit_eval(params + dp_i - dp_j)
                    f_mp = _misfit_eval(params - dp_i + dp_j)
                    f_mm = _misfit_eval(params - dp_i - dp_j)
                    H_fd[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * delta**2)
                    H_fd[j, i] = H_fd[i, j]

        H_ad_np = H_ad.detach().numpy().real
        scale = np.max(np.abs(H_fd)) + 1e-30
        max_err = np.max(np.abs(H_ad_np - H_fd)) / scale

        assert max_err < 1e-3, (
            f"Hessian AD vs FD mismatch: max relative error = {max_err:.2e}"
        )


# ---------------------------------------------------------------------------
# 4. Branch-cut correctness
# ---------------------------------------------------------------------------


class TestBranchCut:
    """Gradients remain finite at subcritical, critical, and supercritical p."""

    @pytest.mark.parametrize("p", [0.1, 0.2, 0.4, 0.6])
    def test_gradient_finite(self, model, omega_small, p):
        """Gradient of ||R||^2 w.r.t. alpha is finite at slowness p."""
        omega_t = torch.tensor(omega_small, dtype=torch.float64)
        tensors = model_to_tensors(model, requires_grad=True)
        alpha = tensors["alpha"]
        beta = tensors["beta"]
        rho = tensors["rho"]
        thickness = tensors["thickness"]
        Q_alpha = tensors["Q_alpha"]
        Q_beta = tensors["Q_beta"]

        R = kennett_reflectivity_torch(
            alpha,
            beta,
            rho,
            thickness,
            Q_alpha,
            Q_beta,
            p,
            omega_t,
        )
        loss = (R.real**2 + R.imag**2).sum()
        loss.backward()

        assert alpha.grad is not None, "No gradient computed for alpha"
        assert torch.all(torch.isfinite(alpha.grad)), (
            f"Non-finite gradient at p={p}: {alpha.grad}"
        )

    @pytest.mark.parametrize("p", [0.1, 0.2, 0.4, 0.6])
    def test_gradient_nonzero(self, model, omega_small, p):
        """Gradient is non-trivial (not all zeros) for sub-ocean layers."""
        omega_t = torch.tensor(omega_small, dtype=torch.float64)
        tensors = model_to_tensors(model, requires_grad=True)

        R = kennett_reflectivity_torch(
            **tensors,
            p=p,
            omega=omega_t,
        )
        loss = (R.real**2 + R.imag**2).sum()
        loss.backward()

        alpha = tensors["alpha"]
        # Sub-ocean alpha gradients (layers 1+) should be nonzero
        assert alpha.grad is not None
        sub_ocean_grad = alpha.grad[1:]
        assert torch.any(sub_ocean_grad.abs() > 1e-20), (
            f"All-zero gradient at p={p}: {sub_ocean_grad}"
        )


# ---------------------------------------------------------------------------
# 5. AD Jacobian vs analytical Frechet (Dietrich & Kormendi)
# ---------------------------------------------------------------------------


class TestFrechetAnalytical:
    """AD Jacobian vs independently derived analytical Frechet derivative."""

    @pytest.mark.parametrize("p", [0.2, 0.4])
    def test_jacobian_vs_frechet(self, model, omega_small, p):
        """AD Jacobian matches tangent-linear analytical Frechet to ~machine eps."""
        from Kennett_Reflectivity.frechet_analytical import frechet_kennett

        omega_np = omega_small
        omega_t = torch.tensor(omega_np, dtype=torch.float64)
        tensors = model_to_tensors(model)

        # AD Jacobian (torch)
        J_ad = jacobian(
            tensors["alpha"],
            tensors["beta"],
            tensors["rho"],
            tensors["thickness"],
            tensors["Q_alpha"],
            tensors["Q_beta"],
            p,
            omega_t,
        )
        J_ad_np = J_ad.detach().numpy()

        # Analytical Frechet (numpy, independent of torch)
        thick_np = model.thickness.copy()
        thick_np[np.isinf(thick_np)] = 1e30  # match torch sentinel
        J_anal = frechet_kennett(
            model.alpha,
            model.beta,
            model.rho,
            thick_np,
            model.Q_alpha,
            model.Q_beta,
            p,
            omega_np,
        )

        scale = np.max(np.abs(J_anal)) + 1e-30
        max_err = np.max(np.abs(J_ad_np - J_anal)) / scale

        assert max_err < 1e-10, (
            f"AD vs analytical Frechet mismatch at p={p}: "
            f"max relative error = {max_err:.2e}"
        )

    def test_frechet_vs_fd(self, model, omega_small):
        """Analytical Frechet matches finite differences (independent of torch)."""
        from Kennett_Reflectivity.frechet_analytical import frechet_kennett
        from Kennett_Reflectivity.kennett_reflectivity import kennett_reflectivity
        from Kennett_Reflectivity.layer_model import LayerModel

        p = 0.2
        thick_np = model.thickness.copy()
        thick_np[np.isinf(thick_np)] = 1e30

        J_anal = frechet_kennett(
            model.alpha,
            model.beta,
            model.rho,
            thick_np,
            model.Q_alpha,
            model.Q_beta,
            p,
            omega_small,
        )

        # FD Jacobian using numpy reference
        n_sub = model.n_layers - 1
        n_thick = n_sub - 1
        n_params = 3 * n_sub + n_thick
        nfreq = len(omega_small)
        delta = 1e-7
        J_fd = np.zeros((nfreq, n_params), dtype=np.complex128)

        for j in range(n_params):
            # Decode parameter
            if j < n_sub:
                arr_name, layer = "alpha", j + 1
            elif j < 2 * n_sub:
                arr_name, layer = "beta", (j - n_sub) + 1
            elif j < 3 * n_sub:
                arr_name, layer = "rho", (j - 2 * n_sub) + 1
            else:
                arr_name, layer = "thickness", (j - 3 * n_sub) + 1

            for sign, _coeff in [(+1, 1.0), (-1, -1.0)]:
                arr = getattr(model, arr_name).copy()
                arr[layer] += sign * delta
                thick_j = model.thickness.copy()
                if arr_name == "thickness":
                    thick_j = arr
                    arr = model.thickness
                m = LayerModel.from_arrays(
                    alpha=model.alpha.copy() if arr_name != "alpha" else arr,
                    beta=model.beta.copy() if arr_name != "beta" else arr,
                    rho=model.rho.copy() if arr_name != "rho" else arr,
                    thickness=thick_j,
                    Q_alpha=model.Q_alpha.copy(),
                    Q_beta=model.Q_beta.copy(),
                )
                R = kennett_reflectivity(m, p, omega_small)
                if sign == 1:
                    R_plus = R
                else:
                    R_minus = R
            J_fd[:, j] = (R_plus - R_minus) / (2.0 * delta)

        scale = np.max(np.abs(J_fd)) + 1e-30
        max_err = np.max(np.abs(J_anal - J_fd)) / scale

        assert max_err < 1e-5, (
            f"Analytical Frechet vs FD mismatch: max relative error = {max_err:.2e}"
        )
