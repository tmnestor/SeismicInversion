"""YAML configuration for neural inference network training and hybrid inversion.

Follows the config pattern established in ``Kennett_Reflectivity.inversion_config``
with dataclass-based configuration, fail-fast validation, and YAML round-trip support.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml


__all__ = [
    "ArchitectureConfig",
    "DataGenConfig",
    "HybridInversionConfig",
    "InferenceNetConfig",
    "TrainingConfig",
    "load_inference_config",
    "save_inference_config",
]


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DataGenConfig:
    """Configuration for training data generation."""

    n_train: int = 50_000
    n_val: int = 5_000
    n_test: int = 5_000
    velocity_perturbation: float = 0.20
    density_perturbation: float = 0.15
    thickness_perturbation: float = 0.25
    p_values: list[float] = field(
        default_factory=lambda: [
            0.05,
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
        ]
    )
    nfreq: int = 64
    seed: int = 42


@dataclass
class ArchitectureConfig:
    """Transformer encoder architecture hyperparameters."""

    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 128
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Training loop hyperparameters."""

    n_epochs: int = 200
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 20
    grad_clip: float = 1.0


@dataclass
class HybridInversionConfig:
    """Configuration for the hybrid Newton-Net inversion pipeline."""

    max_newton_iter: int = 10
    newton_tol: float = 1e-8
    initial_damping: float | None = None
    use_laplace: bool = False


@dataclass
class InferenceNetConfig:
    """Top-level composite configuration for neural inference."""

    data: DataGenConfig = field(default_factory=DataGenConfig)
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    hybrid: HybridInversionConfig = field(default_factory=HybridInversionConfig)
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    data_dir: Path = field(default_factory=lambda: Path("data"))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class ConfigError(Exception):
    """Raised when a config file is invalid."""


def _require(condition: bool, msg: str) -> None:
    """Fail fast with a clear diagnostic if *condition* is False."""
    if not condition:
        raise ConfigError(msg)


def _validate_data(cfg: DataGenConfig) -> None:
    """Validate data generation config."""
    _require(cfg.n_train > 0, f"data.n_train must be > 0, got {cfg.n_train}")
    _require(cfg.n_val > 0, f"data.n_val must be > 0, got {cfg.n_val}")
    _require(cfg.n_test > 0, f"data.n_test must be > 0, got {cfg.n_test}")
    _require(
        0 < cfg.velocity_perturbation < 1,
        f"data.velocity_perturbation must be in (0, 1), got {cfg.velocity_perturbation}",
    )
    _require(
        0 < cfg.density_perturbation < 1,
        f"data.density_perturbation must be in (0, 1), got {cfg.density_perturbation}",
    )
    _require(
        0 < cfg.thickness_perturbation < 1,
        f"data.thickness_perturbation must be in (0, 1), got {cfg.thickness_perturbation}",
    )
    _require(len(cfg.p_values) > 0, "data.p_values must be non-empty")
    for i, p in enumerate(cfg.p_values):
        _require(p > 0, f"data.p_values[{i}] must be > 0, got {p}")
    _require(cfg.nfreq > 0, f"data.nfreq must be > 0, got {cfg.nfreq}")


def _validate_architecture(cfg: ArchitectureConfig) -> None:
    """Validate architecture config."""
    _require(cfg.d_model > 0, f"architecture.d_model must be > 0, got {cfg.d_model}")
    _require(cfg.n_heads > 0, f"architecture.n_heads must be > 0, got {cfg.n_heads}")
    _require(
        cfg.d_model % cfg.n_heads == 0,
        f"architecture.d_model ({cfg.d_model}) must be divisible by "
        f"n_heads ({cfg.n_heads})",
    )
    _require(cfg.n_layers > 0, f"architecture.n_layers must be > 0, got {cfg.n_layers}")
    _require(cfg.d_ff > 0, f"architecture.d_ff must be > 0, got {cfg.d_ff}")
    _require(
        0 <= cfg.dropout < 1,
        f"architecture.dropout must be in [0, 1), got {cfg.dropout}",
    )


def _validate_training(cfg: TrainingConfig) -> None:
    """Validate training config."""
    _require(cfg.n_epochs > 0, f"training.n_epochs must be > 0, got {cfg.n_epochs}")
    _require(
        cfg.batch_size > 0, f"training.batch_size must be > 0, got {cfg.batch_size}"
    )
    _require(
        cfg.learning_rate > 0,
        f"training.learning_rate must be > 0, got {cfg.learning_rate}",
    )
    _require(cfg.patience > 0, f"training.patience must be > 0, got {cfg.patience}")
    _require(cfg.grad_clip > 0, f"training.grad_clip must be > 0, got {cfg.grad_clip}")


def _validate_config(cfg: InferenceNetConfig) -> None:
    """Validate the full config tree."""
    _validate_data(cfg.data)
    _validate_architecture(cfg.architecture)
    _validate_training(cfg.training)


# ---------------------------------------------------------------------------
# Load / Save
# ---------------------------------------------------------------------------


def load_inference_config(path: Path) -> InferenceNetConfig:
    """Load and validate an inference config from a YAML file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Validated InferenceNetConfig.

    Raises:
        ConfigError: If the config is invalid.
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        msg = f"Config file not found: {path.resolve()}"
        raise FileNotFoundError(msg)

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(
        isinstance(raw, dict),
        f"Config file must be a YAML mapping, got {type(raw).__name__}",
    )

    # Data generation
    d = raw.get("data", {})
    data = DataGenConfig(
        n_train=int(d.get("n_train", 50_000)),
        n_val=int(d.get("n_val", 5_000)),
        n_test=int(d.get("n_test", 5_000)),
        velocity_perturbation=float(d.get("velocity_perturbation", 0.20)),
        density_perturbation=float(d.get("density_perturbation", 0.15)),
        thickness_perturbation=float(d.get("thickness_perturbation", 0.25)),
        p_values=d.get(
            "p_values",
            [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
        ),
        nfreq=int(d.get("nfreq", 64)),
        seed=int(d.get("seed", 42)),
    )

    # Architecture
    a = raw.get("architecture", {})
    arch = ArchitectureConfig(
        d_model=int(a.get("d_model", 64)),
        n_heads=int(a.get("n_heads", 4)),
        n_layers=int(a.get("n_layers", 3)),
        d_ff=int(a.get("d_ff", 128)),
        dropout=float(a.get("dropout", 0.1)),
    )

    # Training
    t = raw.get("training", {})
    train = TrainingConfig(
        n_epochs=int(t.get("n_epochs", 200)),
        batch_size=int(t.get("batch_size", 256)),
        learning_rate=float(t.get("learning_rate", 1e-3)),
        weight_decay=float(t.get("weight_decay", 1e-4)),
        patience=int(t.get("patience", 20)),
        grad_clip=float(t.get("grad_clip", 1.0)),
    )

    # Hybrid inversion
    h = raw.get("hybrid", {})
    hybrid = HybridInversionConfig(
        max_newton_iter=int(h.get("max_newton_iter", 10)),
        newton_tol=float(h.get("newton_tol", 1e-8)),
        initial_damping=h.get("initial_damping"),
        use_laplace=bool(h.get("use_laplace", False)),
    )

    cfg = InferenceNetConfig(
        data=data,
        architecture=arch,
        training=train,
        hybrid=hybrid,
        checkpoint_dir=Path(raw.get("checkpoint_dir", "checkpoints")),
        data_dir=Path(raw.get("data_dir", "data")),
    )

    _validate_config(cfg)
    return cfg


def save_inference_config(config: InferenceNetConfig, path: Path) -> None:
    """Serialize an InferenceNetConfig to YAML.

    Args:
        config: Configuration to serialize.
        path: Output YAML path.
    """
    doc: dict[str, object] = {
        "data": {
            "n_train": config.data.n_train,
            "n_val": config.data.n_val,
            "n_test": config.data.n_test,
            "velocity_perturbation": config.data.velocity_perturbation,
            "density_perturbation": config.data.density_perturbation,
            "thickness_perturbation": config.data.thickness_perturbation,
            "p_values": config.data.p_values,
            "nfreq": config.data.nfreq,
            "seed": config.data.seed,
        },
        "architecture": {
            "d_model": config.architecture.d_model,
            "n_heads": config.architecture.n_heads,
            "n_layers": config.architecture.n_layers,
            "d_ff": config.architecture.d_ff,
            "dropout": config.architecture.dropout,
        },
        "training": {
            "n_epochs": config.training.n_epochs,
            "batch_size": config.training.batch_size,
            "learning_rate": config.training.learning_rate,
            "weight_decay": config.training.weight_decay,
            "patience": config.training.patience,
            "grad_clip": config.training.grad_clip,
        },
        "hybrid": {
            "max_newton_iter": config.hybrid.max_newton_iter,
            "newton_tol": config.hybrid.newton_tol,
            "initial_damping": config.hybrid.initial_damping,
            "use_laplace": config.hybrid.use_laplace,
        },
        "checkpoint_dir": str(config.checkpoint_dir),
        "data_dir": str(config.data_dir),
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _inf_representer(dumper: yaml.Dumper, data: float) -> yaml.ScalarNode:
        if math.isinf(data):
            return dumper.represent_scalar("tag:yaml.org,2002:float", ".inf")
        return dumper.represent_float(data)

    dumper = yaml.Dumper
    dumper.add_representer(float, _inf_representer)

    path.write_text(
        yaml.dump(doc, Dumper=dumper, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
