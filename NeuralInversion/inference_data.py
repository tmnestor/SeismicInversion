"""Dataset generation and loading for neural inference training.

Generates training data by sampling perturbed earth models from a prior
distribution centred on the default ocean-crust model, computing forward
reflectivity with the GMM Riccati-sweep solver, and saving to disk as
``.pt`` files for efficient loading.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from GlobalMatrix.gmm_torch import gmm_reflectivity_torch
from Kennett_Reflectivity.kennett_seismogram import default_ocean_crust_model
from Kennett_Reflectivity.kennett_torch import (
    _pack_params,
    _unpack_params,
    model_to_tensors,
)

from .inference_config import DataGenConfig

__all__ = [
    "ReflectivityDataset",
    "generate_training_data",
    "prepare_input",
]

logger = logging.getLogger(__name__)

_FDTYPE = torch.float64


def _perturb_model(
    true_params: np.ndarray,
    n_sub: int,
    rng: np.random.Generator,
    vel_pert: float,
    den_pert: float,
    thick_pert: float,
) -> np.ndarray:
    """Generate a random perturbation of the true parameter vector.

    Args:
        true_params: True packed parameter vector.
        n_sub: Number of sub-ocean layers.
        rng: Random number generator.
        vel_pert: Fractional perturbation for velocities.
        den_pert: Fractional perturbation for densities.
        thick_pert: Fractional perturbation for thicknesses.

    Returns:
        Perturbed parameter vector.
    """
    n_params = len(true_params)
    perturb = np.ones(n_params)
    n_vel = 2 * n_sub  # alpha_sub + beta_sub
    n_thick = n_sub - 1

    perturb[:n_vel] = rng.uniform(1.0 - vel_pert, 1.0 + vel_pert, n_vel)
    perturb[n_vel : n_vel + n_sub] = rng.uniform(1.0 - den_pert, 1.0 + den_pert, n_sub)
    perturb[n_vel + n_sub : n_vel + n_sub + n_thick] = rng.uniform(
        1.0 - thick_pert, 1.0 + thick_pert, n_thick
    )
    return true_params * perturb


def generate_training_data(
    output_dir: Path,
    config: DataGenConfig | None = None,
) -> dict[str, Path]:
    """Generate training, validation, and test datasets.

    Samples earth models from a prior centred on the default ocean-crust model,
    computes forward reflectivity using ``gmm_reflectivity_torch``, and saves
    to ``.pt`` files.

    Args:
        output_dir: Directory for output ``.pt`` files.
        config: Data generation config. Uses defaults if ``None``.

    Returns:
        Dict mapping split names to file paths.
    """
    if config is None:
        from .inference_config import DataGenConfig

        config = DataGenConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    n_p = len(config.p_values)

    # Frequency grid
    T = 64.0
    dw = 2.0 * np.pi / T
    omega = torch.arange(1, config.nfreq + 1, dtype=_FDTYPE) * dw

    splits = {
        "train": config.n_train,
        "val": config.n_val,
        "test": config.n_test,
    }
    paths: dict[str, Path] = {}
    rng = np.random.default_rng(config.seed)

    for split_name, n_samples in splits.items():
        log_params_all = torch.zeros(n_samples, n_params, dtype=torch.float32)
        reflectivity_all = torch.zeros(
            n_samples, n_p, 2 * config.nfreq, dtype=torch.float32
        )

        for i in range(n_samples):
            if (i + 1) % 1000 == 0:
                logger.info("%s: %d/%d", split_name, i + 1, n_samples)

            perturbed = _perturb_model(
                true_params,
                n_sub,
                rng,
                config.velocity_perturbation,
                config.density_perturbation,
                config.thickness_perturbation,
            )
            log_params_all[i] = torch.tensor(np.log(perturbed), dtype=torch.float32)

            # Compute reflectivity for each slowness
            phys_tensor = torch.tensor(perturbed, dtype=_FDTYPE)
            a, b, r, h = _unpack_params(
                phys_tensor, alpha_ref, beta_ref, rho_ref, thickness_ref
            )

            with torch.no_grad():
                for j, p_val in enumerate(config.p_values):
                    R = gmm_reflectivity_torch(
                        a, b, r, h, Q_alpha, Q_beta, p_val, omega
                    )
                    reflectivity_all[i, j, : config.nfreq] = R.real.float()
                    reflectivity_all[i, j, config.nfreq :] = R.imag.float()

        save_path = output_dir / f"{split_name}.pt"
        torch.save(
            {
                "log_params": log_params_all,
                "reflectivity": reflectivity_all,
                "p_values": config.p_values,
                "nfreq": config.nfreq,
                "n_params": n_params,
            },
            save_path,
        )
        paths[split_name] = save_path
        logger.info("Saved %s: %d samples -> %s", split_name, n_samples, save_path)

    return paths


class ReflectivityDataset(Dataset):
    """PyTorch dataset for reflectivity -> log-parameter pairs.

    Each sample contains:
    - ``reflectivity``: shape ``(n_p, 2*nfreq)`` float32, per-sample normalised
    - ``log_params``: shape ``(n_params,)`` float32

    Args:
        path: Path to a ``.pt`` file generated by ``generate_training_data``.
        normalise: Whether to apply per-sample normalisation to reflectivity.
    """

    def __init__(self, path: Path, normalise: bool = True) -> None:
        data = torch.load(path, weights_only=True)
        self.reflectivity: torch.Tensor = data["reflectivity"]
        self.log_params: torch.Tensor = data["log_params"]
        self.p_values: list[float] = data["p_values"]
        self.nfreq: int = data["nfreq"]
        self.normalise = normalise

    def __len__(self) -> int:
        return len(self.log_params)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.reflectivity[idx]  # (n_p, 2*nfreq)
        y = self.log_params[idx]  # (n_params,)

        if self.normalise:
            scale = x.abs().max().clamp(min=1e-10)
            x = x / scale

        return x, y


def prepare_input(
    R_obs: dict[float, torch.Tensor],
    p_values: list[float],
    nfreq: int,
) -> torch.Tensor:
    """Convert observed reflectivity dict to network input tensor.

    Args:
        R_obs: Observed reflectivity keyed by slowness, each ``(nfreq,)`` complex.
        p_values: Ordered slowness values.
        nfreq: Number of frequency bins.

    Returns:
        Input tensor of shape ``(1, n_p, 2*nfreq)`` float32, normalised.
    """
    n_p = len(p_values)
    x = torch.zeros(1, n_p, 2 * nfreq, dtype=torch.float32)
    for j, p_val in enumerate(p_values):
        R = R_obs[p_val]
        x[0, j, :nfreq] = R.real.float()
        x[0, j, nfreq:] = R.imag.float()

    scale = x.abs().max().clamp(min=1e-10)
    x = x / scale
    return x
