"""Tests for generalised source placement in the Riccati solver."""

import numpy as np
import pytest
import torch

from Kennett_Reflectivity.layer_model import LayerModel, vertical_slowness

from .global_matrix import _build_system, _compute_eigenvectors
from .riccati_solver import (
    compute_source_vector,
    riccati_sweep_numpy,
    riccati_sweep_torch,
)


# ---- Fixtures ----


@pytest.fixture
def model_4layer() -> LayerModel:
    """4-layer model: ocean + 2 elastic + half-space."""
    return LayerModel.from_arrays(
        alpha=[1.5, 3.0, 4.5, 6.0],
        beta=[0.0, 1.5, 2.5, 3.5],
        rho=[1.0, 2.5, 2.8, 3.2],
        thickness=[2.0, 1.0, 1.5, np.inf],
        Q_alpha=[20000, 100, 100, 100],
        Q_beta=[1e10, 100, 100, 100],
    )


@pytest.fixture
def omega() -> np.ndarray:
    T = 64.0
    nw = 64
    dw = 2.0 * np.pi / T
    return np.arange(1, nw, dtype=np.float64) * dw


def _model_arrays(model: LayerModel, p: float):
    """Extract arrays needed for low-level solver calls."""
    nlayer = model.n_layers
    s_p = model.complex_slowness_p()
    s_s = model.complex_slowness_s()
    beta_c = model.complex_velocity_s()
    cp = complex(p)
    eta = np.array(
        [vertical_slowness(s_p[i], cp) for i in range(nlayer)],
        dtype=np.complex128,
    )
    neta = np.zeros(nlayer, dtype=np.complex128)
    for i in range(1, nlayer):
        neta[i] = vertical_slowness(s_s[i], cp)
    return nlayer, eta, neta, model.rho, beta_c, model.thickness, cp


# ===== Backward compatibility =====


class TestBackwardCompatibility:
    """source_terms=None gives identical results to the original code."""

    @pytest.mark.parametrize("p", [0.1, 0.3])
    def test_none_source_terms(self, model_4layer, omega, p):
        nlayer, eta, neta, rho, beta_c, thickness, cp = _model_arrays(model_4layer, p)
        E_d, E_u, phase_d, e_d_oc, e_u_oc, e0 = _compute_eigenvectors(
            nlayer, eta, neta, rho, beta_c, thickness, cp, omega
        )

        R_default, U_default = riccati_sweep_numpy(
            nlayer, E_d, E_u, phase_d, e_d_oc, e_u_oc, e0
        )
        R_none, U_none = riccati_sweep_numpy(
            nlayer, E_d, E_u, phase_d, e_d_oc, e_u_oc, e0, source_terms=None
        )

        np.testing.assert_array_equal(R_default, R_none)
        np.testing.assert_array_equal(U_default, U_none)


# ===== Zero source terms =====


class TestZeroSourceTerms:
    """All-zero source_terms gives same result as no source_terms."""

    @pytest.mark.parametrize("p", [0.1, 0.3])
    def test_zero_source(self, model_4layer, omega, p):
        nlayer, eta, neta, rho, beta_c, thickness, cp = _model_arrays(model_4layer, p)
        nfreq = len(omega)
        E_d, E_u, phase_d, e_d_oc, e_u_oc, e0 = _compute_eigenvectors(
            nlayer, eta, neta, rho, beta_c, thickness, cp, omega
        )

        R_default, _ = riccati_sweep_numpy(
            nlayer, E_d, E_u, phase_d, e_d_oc, e_u_oc, e0
        )

        # Source terms with zero vectors at every interface
        zero_src: dict[int, np.ndarray] = {}
        M = nlayer - 2
        for k in range(0, M + 1):
            zero_src[k] = np.zeros((nfreq, 4), dtype=np.complex128)

        R_zero, _ = riccati_sweep_numpy(
            nlayer, E_d, E_u, phase_d, e_d_oc, e_u_oc, e0, source_terms=zero_src
        )

        np.testing.assert_allclose(R_zero, R_default, rtol=1e-12, atol=1e-14)


# ===== Dense vs Riccati with buried source =====


class TestDenseVsRiccatiSource:
    """Dense and Riccati solvers agree for buried source."""

    @pytest.mark.parametrize("source_layer", [1, 2])
    @pytest.mark.parametrize("source_frac", [0.3, 0.7])
    @pytest.mark.parametrize("p", [0.1, 0.3])
    def test_dense_riccati_agree(
        self, model_4layer, omega, p, source_layer, source_frac
    ):
        nlayer, eta, neta, rho, beta_c, thickness, cp = _model_arrays(model_4layer, p)
        nfreq = len(omega)
        E_d, E_u, phase_d, e_d_oc, e_u_oc, e0 = _compute_eigenvectors(
            nlayer, eta, neta, rho, beta_c, thickness, cp, omega
        )

        # Unit vertical-force source (jump in σ_zz)
        S = np.zeros((nfreq, 4), dtype=np.complex128)
        S[:, 2] = 1.0

        source_terms = compute_source_vector(
            S,
            source_frac,
            E_d[source_layer],
            E_u[source_layer],
            eta[source_layer],
            neta[source_layer],
            thickness[source_layer],
            omega,
            source_layer,
        )

        # Riccati solver
        R_riccati, _ = riccati_sweep_numpy(
            nlayer, E_d, E_u, phase_d, e_d_oc, e_u_oc, e0, source_terms=source_terms
        )

        # Dense solver
        G, b_vec, N = _build_system(
            nlayer,
            eta,
            neta,
            rho,
            beta_c,
            thickness,
            cp,
            omega,
            source_terms=source_terms,
        )
        x = np.linalg.solve(G, b_vec[..., np.newaxis])[..., 0]
        R_dense = e0 * x[:, 0]

        mask = np.abs(R_dense) > 1e-15
        if np.any(mask):
            rel_err = np.abs(R_riccati[mask] - R_dense[mask]) / np.abs(R_dense[mask])
            assert np.max(rel_err) < 1e-9, f"Max relative error: {np.max(rel_err):.2e}"

        abs_err = np.abs(R_riccati - R_dense)
        assert np.max(abs_err) < 1e-11, f"Max absolute error: {np.max(abs_err):.2e}"

    def test_source_changes_result(self, model_4layer, omega):
        """Buried source produces a different result from surface-only."""
        p = 0.2
        nlayer, eta, neta, rho, beta_c, thickness, cp = _model_arrays(model_4layer, p)
        nfreq = len(omega)
        E_d, E_u, phase_d, e_d_oc, e_u_oc, e0 = _compute_eigenvectors(
            nlayer, eta, neta, rho, beta_c, thickness, cp, omega
        )

        R_default, _ = riccati_sweep_numpy(
            nlayer, E_d, E_u, phase_d, e_d_oc, e_u_oc, e0
        )

        S = np.zeros((nfreq, 4), dtype=np.complex128)
        S[:, 2] = 1.0
        source_terms = compute_source_vector(
            S, 0.5, E_d[1], E_u[1], eta[1], neta[1], thickness[1], omega, 1
        )

        R_with_source, _ = riccati_sweep_numpy(
            nlayer, E_d, E_u, phase_d, e_d_oc, e_u_oc, e0, source_terms=source_terms
        )

        # Results must differ
        assert not np.allclose(R_default, R_with_source, atol=1e-14)


# ===== PyTorch backward compatibility =====


class TestTorchSourceTerms:
    """PyTorch source_terms path matches NumPy."""

    def test_torch_matches_numpy_with_source(self, model_4layer, omega):
        p = 0.2
        nlayer, eta, neta, rho, beta_c, thickness, cp = _model_arrays(model_4layer, p)
        nfreq = len(omega)
        E_d, E_u, phase_d, e_d_oc, e_u_oc, e0 = _compute_eigenvectors(
            nlayer, eta, neta, rho, beta_c, thickness, cp, omega
        )

        S = np.zeros((nfreq, 4), dtype=np.complex128)
        S[:, 2] = 1.0
        source_terms_np = compute_source_vector(
            S, 0.5, E_d[1], E_u[1], eta[1], neta[1], thickness[1], omega, 1
        )

        # NumPy result
        R_np, _ = riccati_sweep_numpy(
            nlayer,
            E_d,
            E_u,
            phase_d,
            e_d_oc,
            e_u_oc,
            e0,
            source_terms=source_terms_np,
        )

        # Convert to torch
        E_d_t = {k: torch.from_numpy(v) for k, v in E_d.items()}
        E_u_t = {k: torch.from_numpy(v) for k, v in E_u.items()}
        phase_d_t = {k: torch.from_numpy(v) for k, v in phase_d.items()}
        e_d_oc_t = torch.from_numpy(e_d_oc)
        e_u_oc_t = torch.from_numpy(e_u_oc)
        e0_t = torch.from_numpy(e0)
        source_terms_t = {k: torch.from_numpy(v) for k, v in source_terms_np.items()}

        R_torch, _ = riccati_sweep_torch(
            nlayer,
            E_d_t,
            E_u_t,
            phase_d_t,
            e_d_oc_t,
            e_u_oc_t,
            e0_t,
            source_terms=source_terms_t,
        )

        np.testing.assert_allclose(
            R_torch.detach().numpy(), R_np, rtol=1e-10, atol=1e-12
        )


# ===== Many-layer model with buried source =====


class TestManyLayerSource:
    """Dense vs Riccati with source in a many-layer model."""

    @staticmethod
    def _make_model(n_elastic: int = 10) -> LayerModel:
        rng = np.random.default_rng(99)
        n_total = n_elastic + 2
        alpha = np.zeros(n_total)
        beta = np.zeros(n_total)
        rho = np.zeros(n_total)
        thickness = np.zeros(n_total)
        Q_alpha = np.zeros(n_total)
        Q_beta = np.zeros(n_total)

        alpha[0], beta[0], rho[0] = 1.5, 0.0, 1.0
        thickness[0] = 2.0
        Q_alpha[0], Q_beta[0] = 20000, 1e10

        for i in range(1, n_elastic + 1):
            alpha[i] = 2.5 + 0.2 * i + rng.uniform(-0.05, 0.05)
            beta[i] = alpha[i] * 0.5
            rho[i] = 2.2 + 0.05 * i
            thickness[i] = 0.5 + rng.uniform(0, 0.3)
            Q_alpha[i] = Q_beta[i] = 100

        alpha[-1] = alpha[-2] + 0.5
        beta[-1] = beta[-2] + 0.3
        rho[-1] = rho[-2] + 0.2
        thickness[-1] = np.inf
        Q_alpha[-1] = Q_beta[-1] = 100

        return LayerModel.from_arrays(
            alpha=alpha.tolist(),
            beta=beta.tolist(),
            rho=rho.tolist(),
            thickness=thickness.tolist(),
            Q_alpha=Q_alpha.tolist(),
            Q_beta=Q_beta.tolist(),
        )

    @pytest.mark.parametrize("source_layer", [1, 5, 10])
    def test_many_layer_dense_vs_riccati(self, omega, source_layer):
        model = self._make_model(10)
        p = 0.15
        nlayer, eta, neta, rho, beta_c, thickness, cp = _model_arrays(model, p)
        nfreq = len(omega)
        E_d, E_u, phase_d, e_d_oc, e_u_oc, e0 = _compute_eigenvectors(
            nlayer, eta, neta, rho, beta_c, thickness, cp, omega
        )

        S = np.zeros((nfreq, 4), dtype=np.complex128)
        S[:, 3] = 1.0  # unit horizontal-force source

        source_terms = compute_source_vector(
            S,
            0.5,
            E_d[source_layer],
            E_u[source_layer],
            eta[source_layer],
            neta[source_layer],
            thickness[source_layer],
            omega,
            source_layer,
        )

        R_riccati, _ = riccati_sweep_numpy(
            nlayer,
            E_d,
            E_u,
            phase_d,
            e_d_oc,
            e_u_oc,
            e0,
            source_terms=source_terms,
        )

        G, b_vec, N = _build_system(
            nlayer,
            eta,
            neta,
            rho,
            beta_c,
            thickness,
            cp,
            omega,
            source_terms=source_terms,
        )
        x = np.linalg.solve(G, b_vec[..., np.newaxis])[..., 0]
        R_dense = e0 * x[:, 0]

        mask = np.abs(R_dense) > 1e-15
        if np.any(mask):
            rel_err = np.abs(R_riccati[mask] - R_dense[mask]) / np.abs(R_dense[mask])
            assert np.max(rel_err) < 1e-8, f"Max relative error: {np.max(rel_err):.2e}"

        abs_err = np.abs(R_riccati - R_dense)
        assert np.max(abs_err) < 1e-10, f"Max absolute error: {np.max(abs_err):.2e}"
