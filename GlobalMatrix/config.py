"""YAML configuration loader for GMM reflectivity computations.

Provides dataclass-based configuration with validation, serialization,
and round-trip support. Reuses the Kennett_Reflectivity LayerModel.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from Kennett_Reflectivity.layer_model import LayerModel

__all__ = [
    "GMMConfig",
    "OutputConfig",
    "load_config",
    "save_config",
]

_VALID_FORMATS = frozenset({"reflectivity", "comparison"})
_REQUIRED_LAYER_FIELDS = ("alpha", "beta", "rho", "thickness", "Q_alpha", "Q_beta")


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class OutputConfig:
    """Output directory and format selection."""

    directory: Path = field(default_factory=lambda: Path("figures"))
    formats: list[str] = field(default_factory=lambda: ["reflectivity", "comparison"])


@dataclass
class GMMConfig:
    """Complete configuration for a GMM reflectivity computation."""

    model: LayerModel
    layer_names: list[str]
    fixed_layers: list[int]
    p_values: list[float]
    nfreq: int = 256
    free_surface: bool = False
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

    for i, layer_def in enumerate(layers):
        prefix = f"model.layers[{i}]"
        for fld in _REQUIRED_LAYER_FIELDS:
            _require(
                fld in layer_def,
                f"{prefix} missing required field '{fld}'. "
                f"Each layer must have: {', '.join(_REQUIRED_LAYER_FIELDS)}. "
                f"Example: {{alpha: 3.0, beta: 1.5, rho: 3.0, "
                f"thickness: 1.0, Q_alpha: 100, Q_beta: 100}}",
            )

        _require(
            layer_def["alpha"] > 0,
            f"{prefix}.alpha must be > 0, got {layer_def['alpha']}",
        )
        _require(
            layer_def["beta"] >= 0,
            f"{prefix}.beta must be >= 0, got {layer_def['beta']}",
        )
        _require(
            layer_def["rho"] > 0,
            f"{prefix}.rho must be > 0, got {layer_def['rho']}",
        )
        _require(
            layer_def["Q_alpha"] > 0,
            f"{prefix}.Q_alpha must be > 0, got {layer_def['Q_alpha']}",
        )
        _require(
            layer_def["Q_beta"] > 0,
            f"{prefix}.Q_beta must be > 0, got {layer_def['Q_beta']}",
        )

    # First layer must be acoustic ocean
    _require(
        layers[0]["beta"] == 0,
        f"model.layers[0].beta must be 0 (acoustic ocean layer), "
        f"got {layers[0]['beta']}",
    )

    # Last layer must be half-space (infinite thickness)
    last_h = layers[-1]["thickness"]
    _require(
        math.isinf(last_h),
        f"model.layers[{len(layers) - 1}].thickness must be .inf (half-space), "
        f"got {last_h}. Set thickness: .inf for the bottom layer.",
    )


def _validate_p_values(p_values: list[float]) -> None:
    """Validate slowness grid."""
    _require(len(p_values) > 0, "computation.p_values must be non-empty")
    for i, p in enumerate(p_values):
        _require(p > 0, f"computation.p_values[{i}] must be > 0, got {p}")


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


def load_config(path: Path) -> GMMConfig:
    """Load and validate a GMM config from a YAML file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Validated GMMConfig.

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

    layer_names: list[str] = []
    alpha, beta, rho, thickness = [], [], [], []
    q_alpha, q_beta = [], []
    for i, layer_def in enumerate(layers):
        layer_names.append(layer_def.get("name", f"Layer {i}"))
        alpha.append(float(layer_def["alpha"]))
        beta.append(float(layer_def["beta"]))
        rho.append(float(layer_def["rho"]))
        h = float(layer_def["thickness"])
        thickness.append(np.inf if math.isinf(h) else h)
        q_alpha.append(float(layer_def["Q_alpha"]))
        q_beta.append(float(layer_def["Q_beta"]))

    model = LayerModel.from_arrays(
        alpha=alpha,
        beta=beta,
        rho=rho,
        thickness=thickness,
        Q_alpha=q_alpha,
        Q_beta=q_beta,
    )

    # --- Computation ---
    comp = raw.get("computation", {})
    p_values = comp.get("p_values", [0.1, 0.2, 0.3, 0.4, 0.6])
    _validate_p_values(p_values)
    nfreq = int(comp.get("nfreq", 256))
    free_surface = bool(comp.get("free_surface", False))

    # --- Output ---
    out_sec = raw.get("output", {})
    out_dir = Path(out_sec.get("directory", "figures"))
    formats = out_sec.get("formats", ["reflectivity", "comparison"])
    _validate_formats(formats)

    output = OutputConfig(directory=out_dir, formats=formats)

    return GMMConfig(
        model=model,
        layer_names=layer_names,
        fixed_layers=fixed,
        p_values=p_values,
        nfreq=nfreq,
        free_surface=free_surface,
        output=output,
    )


# ---------------------------------------------------------------------------
# Save (for reproducibility)
# ---------------------------------------------------------------------------


def save_config(config: GMMConfig, path: Path) -> None:
    """Serialize a GMMConfig to YAML for reproducibility.

    Args:
        config: Configuration to serialize.
        path: Output YAML path.
    """
    layers = []
    for i in range(config.model.n_layers):
        layer: dict[str, object] = {"name": config.layer_names[i]}
        layer["alpha"] = float(config.model.alpha[i])
        layer["beta"] = float(config.model.beta[i])
        layer["rho"] = float(config.model.rho[i])
        h = float(config.model.thickness[i])
        layer["thickness"] = float("inf") if np.isinf(h) else h
        layer["Q_alpha"] = float(config.model.Q_alpha[i])
        layer["Q_beta"] = float(config.model.Q_beta[i])
        layers.append(layer)

    doc: dict[str, object] = {
        "model": {
            "layers": layers,
            "fixed_layers": config.fixed_layers,
        },
        "computation": {
            "p_values": config.p_values,
            "nfreq": config.nfreq,
            "free_surface": config.free_surface,
        },
        "output": {
            "directory": str(config.output.directory),
            "formats": config.output.formats,
        },
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
