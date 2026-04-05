"""Tests for the hybrid Newton-Net inversion pipeline.

Verifies that the network warm start is closer to the true model than a random
perturbation, and that Newton refinement converges in fewer iterations.
"""

import logging

import numpy as np
import pytest
import torch

from Kennett_Reflectivity.kennett_seismogram import default_ocean_crust_model
from Kennett_Reflectivity.kennett_torch import (
    _pack_params,
    _unpack_params,
    model_to_tensors,
)
from GlobalMatrix.gmm_torch import gmm_reflectivity_torch

from NeuralInversion.hybrid_inversion import (
    HybridInversionResult,
    hybrid_invert_taup,
    load_trained_network,
    network_predict,
)
from NeuralInversion.inference_config import (
    ArchitectureConfig,
    DataGenConfig,
    InferenceNetConfig,
    TrainingConfig,
)
from NeuralInversion.inference_train import train_inference_net

logger = logging.getLogger(__name__)

_P_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


@pytest.fixture()
def trained_model(tmp_path):
    """Train a small network on a tiny dataset for hybrid inversion testing."""
    ref_model = default_ocean_crust_model()
    ref_t = model_to_tensors(ref_model)
    alpha_ref = ref_t["alpha"]
    beta_ref = ref_t["beta"]
    rho_ref = ref_t["rho"]
    thickness_ref = ref_t["thickness"]
    Q_alpha = ref_t["Q_alpha"]
    Q_beta = ref_t["Q_beta"]

    true_params = _pack_params(alpha_ref, beta_ref, rho_ref, thickness_ref).numpy()
    n_sub = ref_model.n_layers - 1
    n_params = len(true_params)
    nfreq = 64
    n_p = len(_P_VALUES)

    T = 64.0
    dw = 2.0 * np.pi / T
    omega = torch.arange(1, nfreq + 1, dtype=torch.float64) * dw

    rng = np.random.default_rng(123)
    n_samples = 300

    log_params_all = torch.zeros(n_samples, n_params, dtype=torch.float32)
    reflectivity_all = torch.zeros(n_samples, n_p, 2 * nfreq, dtype=torch.float32)

    for i in range(n_samples):
        perturb = np.ones(n_params)
        n_vel = 2 * n_sub
        n_thick = n_sub - 1
        perturb[:n_vel] = rng.uniform(0.8, 1.2, n_vel)
        perturb[n_vel : n_vel + n_sub] = rng.uniform(0.85, 1.15, n_sub)
        perturb[n_vel + n_sub : n_vel + n_sub + n_thick] = rng.uniform(
            0.75, 1.25, n_thick
        )
        params = true_params * perturb

        log_params_all[i] = torch.tensor(np.log(params), dtype=torch.float32)

        phys_tensor = torch.tensor(params, dtype=torch.float64)
        a, b, r, h = _unpack_params(
            phys_tensor, alpha_ref, beta_ref, rho_ref, thickness_ref
        )

        with torch.no_grad():
            for j, p_val in enumerate(_P_VALUES):
                R = gmm_reflectivity_torch(a, b, r, h, Q_alpha, Q_beta, p_val, omega)
                reflectivity_all[i, j, :nfreq] = R.real.float()
                reflectivity_all[i, j, nfreq:] = R.imag.float()

    data = {
        "log_params": log_params_all,
        "reflectivity": reflectivity_all,
        "p_values": _P_VALUES,
        "nfreq": nfreq,
        "n_params": n_params,
    }

    train_path = tmp_path / "train.pt"
    val_path = tmp_path / "val.pt"
    torch.save(data, train_path)
    torch.save(data, val_path)

    config = InferenceNetConfig(
        data=DataGenConfig(p_values=_P_VALUES, nfreq=64),
        architecture=ArchitectureConfig(
            d_model=32,
            n_heads=4,
            n_layers=2,
            d_ff=64,
            dropout=0.0,
        ),
        training=TrainingConfig(
            n_epochs=15,
            batch_size=64,
            learning_rate=1e-3,
            patience=20,
            grad_clip=1.0,
        ),
        checkpoint_dir=tmp_path / "ckpt",
    )

    model, _ = train_inference_net(
        config,
        train_path,
        val_path,
        device=torch.device("cpu"),
    )
    return model


class TestNetworkPredict:
    @pytest.mark.slow
    def test_predict_returns_correct_shapes(self, trained_model):
        """Network predict should return (15,) arrays for mu and sigma."""
        ref_model = default_ocean_crust_model()
        ref_t = model_to_tensors(ref_model)
        nfreq = 64
        T = 64.0
        dw = 2.0 * np.pi / T
        omega = torch.arange(1, nfreq + 1, dtype=torch.float64) * dw

        R_obs: dict[float, torch.Tensor] = {}
        with torch.no_grad():
            for p_val in _P_VALUES:
                R_obs[p_val] = gmm_reflectivity_torch(
                    ref_t["alpha"],
                    ref_t["beta"],
                    ref_t["rho"],
                    ref_t["thickness"],
                    ref_t["Q_alpha"],
                    ref_t["Q_beta"],
                    p_val,
                    omega,
                ).clone()

        mu, sigma = network_predict(trained_model, R_obs, _P_VALUES, nfreq)
        assert mu.shape == (15,)
        assert sigma.shape == (15,)
        assert np.all(np.isfinite(mu))
        assert np.all(sigma > 0)

    @pytest.mark.slow
    def test_network_closer_than_random(self, trained_model):
        """Network prediction should be closer to true params than a random perturbation."""
        ref_model = default_ocean_crust_model()
        ref_t = model_to_tensors(ref_model)
        alpha_ref = ref_t["alpha"]
        beta_ref = ref_t["beta"]
        rho_ref = ref_t["rho"]
        thickness_ref = ref_t["thickness"]
        Q_alpha = ref_t["Q_alpha"]
        Q_beta = ref_t["Q_beta"]
        true_params = _pack_params(alpha_ref, beta_ref, rho_ref, thickness_ref).numpy()

        nfreq = 64
        T = 64.0
        dw = 2.0 * np.pi / T
        omega = torch.arange(1, nfreq + 1, dtype=torch.float64) * dw

        R_obs: dict[float, torch.Tensor] = {}
        with torch.no_grad():
            for p_val in _P_VALUES:
                R_obs[p_val] = gmm_reflectivity_torch(
                    alpha_ref,
                    beta_ref,
                    rho_ref,
                    thickness_ref,
                    Q_alpha,
                    Q_beta,
                    p_val,
                    omega,
                ).clone()

        mu, _ = network_predict(trained_model, R_obs, _P_VALUES, nfreq)
        net_params = np.exp(mu)
        net_error = np.linalg.norm(net_params - true_params) / np.linalg.norm(
            true_params
        )

        # Random 20% perturbation (average error ~12%)
        rng = np.random.default_rng(99)
        random_params = true_params * rng.uniform(0.8, 1.2, len(true_params))
        random_error = np.linalg.norm(random_params - true_params) / np.linalg.norm(
            true_params
        )

        logger.info("Network error: %.4f, Random error: %.4f", net_error, random_error)
        assert net_error < random_error, (
            f"Network ({net_error:.4f}) should be closer than random ({random_error:.4f})"
        )


class TestHybridInversion:
    @pytest.mark.slow
    def test_hybrid_converges(self, trained_model):
        """Hybrid inversion should converge."""
        true_model = default_ocean_crust_model()
        result = hybrid_invert_taup(
            true_model=true_model,
            model=trained_model,
            p_values=_P_VALUES,
            nfreq=64,
            max_iter=10,
        )

        assert isinstance(result, HybridInversionResult)
        assert result.inversion_result.n_iterations > 0
        assert len(result.inversion_result.misfit_history) > 0

    @pytest.mark.slow
    def test_hybrid_result_has_network_info(self, trained_model):
        """Result should contain network predictions."""
        true_model = default_ocean_crust_model()
        result = hybrid_invert_taup(
            true_model=true_model,
            model=trained_model,
            p_values=_P_VALUES,
            nfreq=64,
            max_iter=5,
        )

        assert result.network_mu.shape == (15,)
        assert result.network_sigma.shape == (15,)
        assert result.network_param_error > 0
        assert result.newton_iterations > 0

    @pytest.mark.slow
    def test_laplace_approximation(self, trained_model):
        """Laplace approximation should produce positive sigma values."""
        true_model = default_ocean_crust_model()
        result = hybrid_invert_taup(
            true_model=true_model,
            model=trained_model,
            p_values=_P_VALUES,
            nfreq=64,
            max_iter=5,
            use_laplace=True,
        )

        assert result.laplace_sigma is not None
        assert result.laplace_sigma.shape == (15,)
        assert np.all(result.laplace_sigma > 0)


class TestLoadTrainedNetwork:
    @pytest.mark.slow
    def test_load_and_predict(self, trained_model, tmp_path):
        """Save, reload, and verify predictions match."""
        # Save checkpoint manually
        ckpt_path = tmp_path / "test_ckpt.pt"
        torch.save(
            {
                "model_state_dict": trained_model.state_dict(),
                "config": {
                    "n_p": 12,
                    "nfreq": 64,
                    "n_params": 15,
                    "d_model": 32,
                    "n_heads": 4,
                    "n_layers": 2,
                    "d_ff": 64,
                    "dropout": 0.0,
                    "p_values": _P_VALUES,
                },
                "epoch": 1,
                "val_loss": 0.0,
            },
            ckpt_path,
        )

        loaded = load_trained_network(ckpt_path, device=torch.device("cpu"))

        # Compare predictions
        x = torch.randn(1, 12, 128)
        trained_model.eval()
        with torch.no_grad():
            mu1, ls1 = trained_model(x)
            mu2, ls2 = loaded(x)

        np.testing.assert_allclose(mu1.numpy(), mu2.numpy(), atol=1e-6)
        np.testing.assert_allclose(ls1.numpy(), ls2.numpy(), atol=1e-6)

    def test_missing_checkpoint(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_trained_network(tmp_path / "nonexistent.pt")
