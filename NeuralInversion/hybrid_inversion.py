"""Hybrid Newton-Net inversion pipeline.

Combines a neural network warm start with the existing Newton-Levenberg-Marquardt
solver for rapid convergence.  The network provides both a point estimate (mu)
and calibrated uncertainty (sigma) for each earth model parameter.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

from GlobalMatrix.gmm_torch import gmm_reflectivity_torch
from GlobalMatrix.taup_inversion import (
    _eval_misfit_only,
    taup_misfit_grad_hessian,
)
from Kennett_Reflectivity.kennett_torch import (
    _pack_params,
    _unpack_params,
    model_to_tensors,
)
from Kennett_Reflectivity.layer_model import LayerModel
from Kennett_Reflectivity.taup_inversion import (
    InversionResult,
    _tensors_to_layer_model,
    newton_lm_step,
)

from .inference_data import prepare_input
from .inference_net import SeismicInferenceNet

__all__ = [
    "HybridInversionResult",
    "hybrid_invert_taup",
    "load_trained_network",
    "network_predict",
]

logger = logging.getLogger(__name__)

_FDTYPE = torch.float64


@dataclass
class HybridInversionResult:
    """Result from the hybrid Newton-Net inversion pipeline.

    Extends the standard inversion result with network predictions and
    comparison metrics.
    """

    inversion_result: InversionResult
    network_mu: np.ndarray  # log-space mean from network, shape (n_params,)
    network_sigma: np.ndarray  # log-space std from network, shape (n_params,)
    newton_iterations: int = 0
    network_param_error: float = 0.0  # relative error of network prediction
    laplace_sigma: np.ndarray | None = None  # from inverse Hessian at solution


def load_trained_network(
    checkpoint_path: Path,
    device: torch.device | None = None,
) -> SeismicInferenceNet:
    """Load a trained SeismicInferenceNet from a checkpoint.

    Args:
        checkpoint_path: Path to the ``.pt`` checkpoint file.
        device: Device to load the model onto.

    Returns:
        Trained model in eval mode.

    Raises:
        FileNotFoundError: If checkpoint does not exist.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        msg = f"Checkpoint not found: {checkpoint_path.resolve()}"
        raise FileNotFoundError(msg)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg = ckpt["config"]

    model = SeismicInferenceNet(
        n_p=cfg["n_p"],
        nfreq=cfg["nfreq"],
        n_params=cfg["n_params"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
        p_values=cfg["p_values"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def network_predict(
    model: SeismicInferenceNet,
    R_obs: dict[float, torch.Tensor],
    p_values: list[float],
    nfreq: int,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a single forward pass to get mu and sigma predictions.

    Args:
        model: Trained inference network.
        R_obs: Observed reflectivity keyed by slowness.
        p_values: Ordered slowness values.
        nfreq: Number of frequency bins.
        device: Device for inference.

    Returns:
        ``(mu, sigma)`` as numpy arrays, shape ``(n_params,)``, in log-space.
    """
    if device is None:
        device = next(model.parameters()).device

    x = prepare_input(R_obs, p_values, nfreq).to(device)

    model.eval()
    with torch.no_grad():
        mu, log_sigma = model(x)

    mu_np = mu[0].cpu().numpy().astype(np.float64)
    sigma_np = np.exp(log_sigma[0].cpu().numpy().astype(np.float64))
    return mu_np, sigma_np


def hybrid_invert_taup(
    true_model: LayerModel,
    model: SeismicInferenceNet,
    p_values: list[float],
    nfreq: int = 64,
    max_iter: int = 10,
    tol: float = 1e-8,
    initial_damping: float | None = None,
    use_laplace: bool = False,
) -> HybridInversionResult:
    """Hybrid inversion: network warm start + Newton-LM refinement.

    1. Generate synthetic observed data from *true_model* using GMM.
    2. Feed R_obs through the trained network to get mu, sigma.
    3. Use mu as warm start for Newton-LM with exact Hessian.
    4. Optionally compute Laplace approximation at the Newton solution.

    Args:
        true_model: True earth model (generates synthetic observed data).
        model: Trained inference network.
        p_values: Slowness grid (s/km).
        nfreq: Number of frequencies.
        max_iter: Maximum Newton iterations.
        tol: Convergence tolerance.
        initial_damping: Fixed initial LM damping (auto if ``None``).
        use_laplace: Whether to compute Laplace sigma from inverse Hessian.

    Returns:
        HybridInversionResult with network predictions and Newton refinement.
    """
    # True model tensors
    true_t = model_to_tensors(true_model)
    alpha_ref = true_t["alpha"]
    beta_ref = true_t["beta"]
    rho_ref = true_t["rho"]
    thickness_ref = true_t["thickness"]
    Q_alpha = true_t["Q_alpha"]
    Q_beta = true_t["Q_beta"]

    # Frequency grid
    T = 64.0
    dw = 2.0 * np.pi / T
    omega = torch.arange(1, nfreq + 1, dtype=_FDTYPE) * dw

    # Synthetic observed data using GMM
    R_obs: dict[float, torch.Tensor] = {}
    with torch.no_grad():
        for p_val in p_values:
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

    # True parameter vector
    true_params = _pack_params(alpha_ref, beta_ref, rho_ref, thickness_ref).numpy()

    # --- Step 1: Network prediction ---
    net_mu, net_sigma = network_predict(model, R_obs, p_values, nfreq)
    net_param_err = float(
        np.linalg.norm(np.exp(net_mu) - true_params) / np.linalg.norm(true_params)
    )
    logger.info("Network prediction: param_error=%.4e", net_param_err)

    # Build initial model from network prediction
    init_params = np.exp(net_mu)
    init_pt = torch.tensor(init_params, dtype=_FDTYPE)
    a_i, b_i, r_i, h_i = _unpack_params(
        init_pt, alpha_ref, beta_ref, rho_ref, thickness_ref
    )
    initial_model = _tensors_to_layer_model(a_i, b_i, r_i, h_i, true_model)

    # --- Step 2: Newton-LM refinement ---
    log_x = net_mu.copy()

    misfit_history: list[float] = []
    grad_norm_history: list[float] = []
    param_error_history: list[float] = []

    common_args = (
        alpha_ref,
        beta_ref,
        rho_ref,
        thickness_ref,
        Q_alpha,
        Q_beta,
        p_values,
        omega,
        R_obs,
    )

    lam = initial_damping
    last_H = None

    for iteration in range(max_iter):
        chi, g, H = taup_misfit_grad_hessian(log_x, *common_args)
        last_H = H

        grad_norm = float(np.linalg.norm(g))
        param_err = float(
            np.linalg.norm(np.exp(log_x) - true_params) / np.linalg.norm(true_params)
        )

        misfit_history.append(chi)
        grad_norm_history.append(grad_norm)
        param_error_history.append(param_err)

        logger.info(
            "iter %3d  chi=%.4e  ||g||=%.4e  err=%.4e",
            iteration,
            chi,
            grad_norm,
            param_err,
        )

        # Initialise damping
        if lam is None:
            lam = 0.01 * max(float(np.abs(np.diag(H)).max()), 1e-10)

        # Convergence checks
        if grad_norm < tol:
            logger.info("Converged: gradient norm < tol")
            break
        if len(misfit_history) > 1:
            rel_change = abs(misfit_history[-2] - chi) / max(abs(chi), 1e-30)
            if rel_change < tol:
                logger.info("Converged: relative misfit change < tol")
                break

        # Newton-LM step with adaptive damping
        accepted = False
        for _ in range(10):
            dx = newton_lm_step(g, H, lam)
            trial_x = log_x + dx
            trial_chi = _eval_misfit_only(trial_x, *common_args)

            if trial_chi < chi:
                log_x = trial_x
                lam = max(lam / 2.0, 1e-15)
                accepted = True
                break
            lam *= 5.0

        if not accepted:
            logger.warning("Line search failed at iteration %d", iteration)
            break

    # Build recovered model
    recovered_params = np.exp(log_x)
    rec_pt = torch.tensor(recovered_params, dtype=_FDTYPE)
    a_r, b_r, r_r, h_r = _unpack_params(
        rec_pt, alpha_ref, beta_ref, rho_ref, thickness_ref
    )
    recovered_model = _tensors_to_layer_model(a_r, b_r, r_r, h_r, true_model)

    converged = bool(param_error_history and param_error_history[-1] < 0.01)

    inv_result = InversionResult(
        true_model=true_model,
        initial_model=initial_model,
        recovered_model=recovered_model,
        misfit_history=misfit_history,
        grad_norm_history=grad_norm_history,
        param_error_history=param_error_history,
        n_iterations=len(misfit_history),
        converged=converged,
    )

    # --- Step 3: Optional Laplace approximation ---
    laplace_sigma = None
    if use_laplace and last_H is not None:
        try:
            H_inv = np.linalg.inv(last_H)
            laplace_sigma = np.sqrt(np.abs(np.diag(H_inv)))
        except np.linalg.LinAlgError:
            logger.warning("Hessian singular — Laplace approximation unavailable")

    return HybridInversionResult(
        inversion_result=inv_result,
        network_mu=net_mu,
        network_sigma=net_sigma,
        newton_iterations=len(misfit_history),
        network_param_error=net_param_err,
        laplace_sigma=laplace_sigma,
    )
