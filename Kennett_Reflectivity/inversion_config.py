"""YAML configuration loader for tau-p inversion experiments.

Provides dataclass-based configuration with validation, serialization,
and round-trip support for reproducible inversion runs.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .layer_model import LayerModel

__all__ = [
    "InversionConfig",
    "OutputConfig",
    "TraceDisplayConfig",
    "load_config",
    "save_config",
]

_VALID_FORMATS = frozenset({"table", "profiles", "traces", "convergence"})
_REQUIRED_LAYER_FIELDS = ("alpha", "beta", "rho", "thickness", "Q_alpha", "Q_beta")


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TraceDisplayConfig:
    """Display settings for tau-p trace comparison plots."""

    t_max: float = 15.0
    nw: int = 1024


@dataclass
class OutputConfig:
    """Output directory and format selection."""

    directory: Path = field(default_factory=lambda: Path("figures"))
    formats: list[str] = field(
        default_factory=lambda: ["table", "profiles", "traces", "convergence"]
    )
    trace_display: TraceDisplayConfig = field(default_factory=TraceDisplayConfig)


@dataclass
class InversionConfig:
    """Complete configuration for a tau-p inversion run."""

    true_model: LayerModel
    layer_names: list[str]
    fixed_layers: list[int]
    p_values: list[float]
    nfreq: int = 64
    perturbation: float = 0.15
    max_iter: int = 100
    seed: int = 42
    tol: float = 1e-8
    output: OutputConfig = field(default_factory=OutputConfig)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


class ConfigError(Exception):
    """Raised when a config file is invalid."""


def _require(condition: bool, msg: str) -> None:
    """Fail fast with a clear diagnostic if *condition* is False."""
    if not condition:
        raise ConfigError(msg)


def _validate_layers(layers: list[dict]) -> None:
    """Validate layer definitions from parsed YAML."""
    _require(
        len(layers) >= 2,
        f"model.layers must have >= 2 layers, got {len(layers)}. "
        "A valid model needs at least one finite layer and a half-space.",
    )

    for i, layer in enumerate(layers):
        prefix = f"model.layers[{i}]"
        for fld in _REQUIRED_LAYER_FIELDS:
            _require(
                fld in layer,
                f"{prefix} missing required field '{fld}'. "
                f"Each layer must have: {', '.join(_REQUIRED_LAYER_FIELDS)}. "
                f"Example: {{alpha: 3.0, beta: 1.5, rho: 3.0, thickness: 1.0, Q_alpha: 100, Q_beta: 100}}",
            )

        _require(
            layer["alpha"] > 0,
            f"{prefix}.alpha must be > 0, got {layer['alpha']}",
        )
        _require(
            layer["beta"] >= 0,
            f"{prefix}.beta must be >= 0, got {layer['beta']}",
        )
        _require(
            layer["rho"] > 0,
            f"{prefix}.rho must be > 0, got {layer['rho']}",
        )
        _require(
            layer["Q_alpha"] > 0,
            f"{prefix}.Q_alpha must be > 0, got {layer['Q_alpha']}",
        )
        _require(
            layer["Q_beta"] > 0,
            f"{prefix}.Q_beta must be > 0, got {layer['Q_beta']}",
        )

    # First layer must be acoustic ocean
    _require(
        layers[0]["beta"] == 0,
        f"model.layers[0].beta must be 0 (acoustic ocean layer), got {layers[0]['beta']}",
    )

    # Last layer must be half-space (infinite thickness)
    last_h = layers[-1]["thickness"]
    _require(
        math.isinf(last_h),
        f"model.layers[{len(layers) - 1}].thickness must be .inf (half-space), "
        f"got {last_h}. Set thickness: .inf for the bottom layer.",
    )


def _validate_fixed_layers(fixed: list[int]) -> None:
    """Validate that fixed_layers is [0]."""
    _require(
        fixed == [0],
        f"model.fixed_layers must be [0] (only ocean layer fixed is supported), "
        f"got {fixed}",
    )


def _validate_p_values(p_values: list[float]) -> None:
    """Validate slowness grid."""
    _require(
        len(p_values) > 0,
        "inversion.p_values must be non-empty",
    )
    for i, p in enumerate(p_values):
        _require(
            p > 0,
            f"inversion.p_values[{i}] must be > 0, got {p}",
        )


def _validate_formats(formats: list[str]) -> None:
    """Validate output format names."""
    for fmt in formats:
        _require(
            fmt in _VALID_FORMATS,
            f"output.formats contains unknown format '{fmt}'. "
            f"Valid formats: {', '.join(sorted(_VALID_FORMATS))}",
        )


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_config(path: Path) -> InversionConfig:
    """Load and validate an inversion config from a YAML file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Validated InversionConfig ready for use with ``invert_taup``.

    Raises:
        ConfigError: If the config is invalid (with actionable diagnostics).
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        msg = f"Config file not found: {path.resolve()}"
        raise FileNotFoundError(msg)

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(
        isinstance(raw, dict) and "model" in raw,
        f"Config file must contain a 'model' key at top level. "
        f"Got keys: {list(raw.keys()) if isinstance(raw, dict) else type(raw).__name__}. "
        f"Example:\n  model:\n    layers:\n      - {{alpha: 1.5, beta: 0.0, ...}}",
    )

    # --- Model ---
    model_sec = raw["model"]
    _require(
        isinstance(model_sec, dict) and "layers" in model_sec,
        "model section must contain a 'layers' list",
    )
    layers = model_sec["layers"]
    _validate_layers(layers)

    fixed = model_sec.get("fixed_layers", [0])
    _validate_fixed_layers(fixed)

    # Build LayerModel
    layer_names: list[str] = []
    alpha, beta, rho, thickness = [], [], [], []
    Q_alpha, Q_beta = [], []
    for i, layer in enumerate(layers):
        layer_names.append(layer.get("name", f"Layer {i}"))
        alpha.append(float(layer["alpha"]))
        beta.append(float(layer["beta"]))
        rho.append(float(layer["rho"]))
        h = float(layer["thickness"])
        thickness.append(np.inf if math.isinf(h) else h)
        Q_alpha.append(float(layer["Q_alpha"]))
        Q_beta.append(float(layer["Q_beta"]))

    true_model = LayerModel.from_arrays(
        alpha=alpha,
        beta=beta,
        rho=rho,
        thickness=thickness,
        Q_alpha=Q_alpha,
        Q_beta=Q_beta,
    )

    # --- Inversion ---
    inv = raw.get("inversion", {})
    p_values = inv.get(
        "p_values",
        [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
    )
    _validate_p_values(p_values)

    nfreq = int(inv.get("nfreq", 64))
    perturbation = float(inv.get("perturbation", 0.15))
    max_iter = int(inv.get("max_iter", 100))
    seed = int(inv.get("seed", 42))
    tol = float(inv.get("tol", 1e-8))

    # --- Output ---
    out_sec = raw.get("output", {})
    out_dir = Path(out_sec.get("directory", "figures"))
    formats = out_sec.get("formats", ["table", "profiles", "traces", "convergence"])
    _validate_formats(formats)

    td = out_sec.get("trace_display", {})
    trace_display = TraceDisplayConfig(
        t_max=float(td.get("t_max", 15.0)),
        nw=int(td.get("nw", 1024)),
    )

    output = OutputConfig(
        directory=out_dir,
        formats=formats,
        trace_display=trace_display,
    )

    return InversionConfig(
        true_model=true_model,
        layer_names=layer_names,
        fixed_layers=fixed,
        p_values=p_values,
        nfreq=nfreq,
        perturbation=perturbation,
        max_iter=max_iter,
        seed=seed,
        tol=tol,
        output=output,
    )


# ---------------------------------------------------------------------------
# Save (for reproducibility)
# ---------------------------------------------------------------------------


def save_config(config: InversionConfig, path: Path) -> None:
    """Serialize an InversionConfig to YAML for reproducibility.

    The saved file can be loaded back with ``load_config`` to reproduce
    the exact same inversion run.

    Args:
        config: Configuration to serialize.
        path: Output YAML path.
    """
    layers = []
    for i in range(config.true_model.n_layers):
        layer: dict[str, object] = {"name": config.layer_names[i]}
        layer["alpha"] = float(config.true_model.alpha[i])
        layer["beta"] = float(config.true_model.beta[i])
        layer["rho"] = float(config.true_model.rho[i])
        h = float(config.true_model.thickness[i])
        layer["thickness"] = float("inf") if np.isinf(h) else h
        layer["Q_alpha"] = float(config.true_model.Q_alpha[i])
        layer["Q_beta"] = float(config.true_model.Q_beta[i])
        layers.append(layer)

    doc: dict[str, object] = {
        "model": {
            "layers": layers,
            "fixed_layers": config.fixed_layers,
        },
        "inversion": {
            "p_values": config.p_values,
            "nfreq": config.nfreq,
            "perturbation": config.perturbation,
            "max_iter": config.max_iter,
            "seed": config.seed,
            "tol": config.tol,
        },
        "output": {
            "directory": str(config.output.directory),
            "formats": config.output.formats,
            "trace_display": {
                "t_max": config.output.trace_display.t_max,
                "nw": config.output.trace_display.nw,
            },
        },
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Custom representer for inf (YAML .inf)
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
