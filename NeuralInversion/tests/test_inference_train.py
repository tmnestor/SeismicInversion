"""Tests for the inference training loop.

Trains for a small number of epochs on a tiny synthetic dataset and
verifies that the loss decreases.
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

from NeuralInversion.inference_config import (
    ArchitectureConfig,
    DataGenConfig,
    InferenceNetConfig,
    TrainingConfig,
)
from NeuralInversion.inference_train import evaluate_inference_net, train_inference_net

logger = logging.getLogger(__name__)

_P_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


@pytest.fixture()
def tiny_dataset(tmp_path):
    """Generate a tiny dataset (200 samples) for quick training tests."""
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

    rng = np.random.default_rng(42)
    n_samples = 200

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
    test_path = tmp_path / "test.pt"

    # Use same data for train/val/test in this tiny test
    torch.save(data, train_path)
    torch.save(data, val_path)
    torch.save(data, test_path)

    return train_path, val_path, test_path


class TestTrainingLoop:
    @pytest.mark.slow
    def test_loss_decreases(self, tiny_dataset, tmp_path):
        """Train for 10 epochs on 200 samples and verify loss decreases."""
        train_path, val_path, test_path = tiny_dataset

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
                n_epochs=10,
                batch_size=32,
                learning_rate=1e-3,
                patience=20,
                grad_clip=1.0,
            ),
            checkpoint_dir=tmp_path / "ckpt",
        )

        model, history = train_inference_net(
            config,
            train_path,
            val_path,
            device=torch.device("cpu"),
        )

        # Loss should decrease over 10 epochs
        assert history["train_loss"][-1] < history["train_loss"][0], (
            f"Training loss did not decrease: {history['train_loss'][0]:.4f} -> "
            f"{history['train_loss'][-1]:.4f}"
        )

    @pytest.mark.slow
    def test_checkpoint_saved(self, tiny_dataset, tmp_path):
        """Verify that a best model checkpoint is saved."""
        train_path, val_path, test_path = tiny_dataset

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
                n_epochs=5,
                batch_size=32,
                learning_rate=1e-3,
                patience=20,
                grad_clip=1.0,
            ),
            checkpoint_dir=tmp_path / "ckpt",
        )

        train_inference_net(config, train_path, val_path, device=torch.device("cpu"))
        assert (tmp_path / "ckpt" / "best_model.pt").exists()


class TestEvaluation:
    @pytest.mark.slow
    def test_evaluate_returns_metrics(self, tiny_dataset, tmp_path):
        """Evaluate on test set and check metric keys."""
        train_path, val_path, test_path = tiny_dataset

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
                n_epochs=5,
                batch_size=32,
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

        metrics = evaluate_inference_net(
            model,
            test_path,
            device=torch.device("cpu"),
            batch_size=32,
        )

        assert "nll" in metrics
        assert "mean_abs_error" in metrics
        assert "calibration_2sigma" in metrics
        assert 0 <= metrics["calibration_2sigma"] <= 1
