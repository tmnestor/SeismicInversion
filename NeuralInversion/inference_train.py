"""Training loop and evaluation for the seismic inference network.

Provides ``train_inference_net`` (AdamW + CosineAnnealingLR with early stopping)
and ``evaluate_inference_net`` (test NLL, mean error, calibration metrics).
"""

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .inference_config import InferenceNetConfig
from .inference_data import ReflectivityDataset
from .inference_net import SeismicInferenceNet, gaussian_nll_loss

__all__ = [
    "evaluate_inference_net",
    "train_inference_net",
]

logger = logging.getLogger(__name__)


def train_inference_net(
    config: InferenceNetConfig,
    train_path: Path,
    val_path: Path,
    device: torch.device | None = None,
) -> tuple[SeismicInferenceNet, dict[str, list[float]]]:
    """Train the inference network with AdamW + cosine annealing + early stopping.

    Args:
        config: Full inference config.
        train_path: Path to training ``.pt`` file.
        val_path: Path to validation ``.pt`` file.
        device: Device to train on (default: auto-detect).

    Returns:
        ``(model, history)`` where *history* has keys ``train_loss``,
        ``val_loss``, ``learning_rate``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets
    train_ds = ReflectivityDataset(train_path)
    val_ds = ReflectivityDataset(val_path)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
    )

    # Model
    arch = config.architecture
    model = SeismicInferenceNet(
        n_p=len(config.data.p_values),
        nfreq=config.data.nfreq,
        n_params=15,
        d_model=arch.d_model,
        n_heads=arch.n_heads,
        n_layers=arch.n_layers,
        d_ff=arch.d_ff,
        dropout=arch.dropout,
        p_values=config.data.p_values,
    ).to(device)

    # Optimiser + scheduler
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max=config.training.n_epochs,
    )

    # Training loop
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
    }
    best_val_loss = float("inf")
    patience_counter = 0

    checkpoint_dir = config.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.training.n_epochs):
        # Train
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            mu, log_sigma = model(x_batch)
            loss = gaussian_nll_loss(mu, log_sigma, y_batch)

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.training.grad_clip
            )
            optimiser.step()

            train_loss_sum += loss.item() * x_batch.size(0)
            train_count += x_batch.size(0)

        train_loss = train_loss_sum / max(train_count, 1)

        # Validate
        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                mu, log_sigma = model(x_batch)
                loss = gaussian_nll_loss(mu, log_sigma, y_batch)

                val_loss_sum += loss.item() * x_batch.size(0)
                val_count += x_batch.size(0)

        val_loss = val_loss_sum / max(val_count, 1)
        lr = optimiser.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rate"].append(lr)

        scheduler.step()

        logger.info(
            "epoch %3d  train=%.4f  val=%.4f  lr=%.2e",
            epoch + 1,
            train_loss,
            val_loss,
            lr,
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "n_p": len(config.data.p_values),
                        "nfreq": config.data.nfreq,
                        "n_params": 15,
                        "d_model": arch.d_model,
                        "n_heads": arch.n_heads,
                        "n_layers": arch.n_layers,
                        "d_ff": arch.d_ff,
                        "dropout": arch.dropout,
                        "p_values": config.data.p_values,
                    },
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                },
                checkpoint_dir / "best_model.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= config.training.patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    # Load best model
    best = torch.load(checkpoint_dir / "best_model.pt", weights_only=True)
    model.load_state_dict(best["model_state_dict"])
    logger.info(
        "Best model from epoch %d (val_loss=%.4f)",
        best["epoch"],
        best["val_loss"],
    )

    return model, history


def evaluate_inference_net(
    model: SeismicInferenceNet,
    test_path: Path,
    device: torch.device | None = None,
    batch_size: int = 256,
) -> dict[str, float]:
    """Evaluate the trained network on a test set.

    Args:
        model: Trained inference network.
        test_path: Path to test ``.pt`` file.
        device: Device for evaluation.
        batch_size: Batch size for evaluation.

    Returns:
        Dict with keys: ``nll`` (test NLL), ``mean_abs_error`` (mean absolute
        error in log-space), ``calibration_2sigma`` (fraction of true parameters
        within 2-sigma of prediction).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = ReflectivityDataset(test_path)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model.eval()
    model.to(device)

    total_nll = 0.0
    total_abs_err = 0.0
    total_within_2sigma = 0
    total_params = 0
    total_count = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            mu, log_sigma = model(x_batch)
            nll = gaussian_nll_loss(mu, log_sigma, y_batch)
            total_nll += nll.item() * x_batch.size(0)

            abs_err = (mu - y_batch).abs()
            total_abs_err += abs_err.sum().item()

            sigma = torch.exp(log_sigma)
            within_2sigma = (abs_err < 2.0 * sigma).sum().item()
            total_within_2sigma += within_2sigma
            total_params += y_batch.numel()
            total_count += x_batch.size(0)

    n_params_per_sample = 15
    return {
        "nll": total_nll / max(total_count, 1),
        "mean_abs_error": total_abs_err / max(total_params, 1),
        "calibration_2sigma": total_within_2sigma / max(total_params, 1),
    }
