"""Helper module for the at-scale heterogeneous-voxel τ-p demo.

Drives a 20-layer oceanic-crust model through the heterogeneous-voxel
block Foldy-Lax inner solver and the Riccati interlayer-MS outer sweep
to produce τ-p wiggle gathers for a full-model and reference-medium
reflectivity.

Pipeline (one realisation):

    YAML → LayerModel → sample per-voxel MaterialContrasts per elastic
    layer → freq-batched ``solve_slab_foldy_lax_freq_het`` per elastic
    layer → ``interlayer_ms_reflectivity_9x9`` per ω → Ricker × IFFT →
    τ-p wiggle gathers + |R(ω,p)| heatmap.

Design notes:
    * Units are km/s, g/cm³, km, GPa (same as the existing test suite).
    * The cube ordering produced by :func:`cluster_from_slab` is
      ``(iz outer, ix middle, iy inner)``; this module's voxel sampler
      follows the same order so that the ``contrasts_per_cube`` list
      consumed by :func:`solve_slab_foldy_lax_freq_het` indexes each
      voxel consistently with the solver.
    * The water layer (layer 0) and the mantle half-space (last layer)
      never carry scatterers.  Scatterer interfaces run ``1..M`` where
      ``M = n_layers - 2``.
    * Fail-fast YAML validation raises :class:`ValueError` with an
      actionable message for invalid thicknesses, ``p_soft`` out of
      ``[0, 1]``, overrides keyed to an invalid elastic layer, or
      negative standard deviations.
"""

import hashlib
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from cubic_scattering import MaterialContrast, ReferenceMedium
from Kennett_Reflectivity.layer_model import LayerModel
from Kennett_Reflectivity.source import ricker_spectrum

from .block_riccati_cluster import (
    solve_slab_foldy_lax_freq_het,
    solve_slab_foldy_lax_single_freq_het_taup,
)
from .interlayer_ms import (
    ScattererSlab9x9,
    ScattererSlab9x9MultiP,
    interlayer_ms_reflectivity_9x9,
    interlayer_ms_reflectivity_9x9_multi_p,
)

__all__ = [
    "ScaleDemoConfig",
    "build_oceanic_crust_model",
    "compute_R_omega_p_stack",
    "compute_layer_composite_tmatrices_het",
    "load_cached_tmatrices",
    "load_scale_demo_config",
    "plot_r_omega_p_heatmap",
    "plot_taup_wiggle_gather",
    "sample_voxel_contrasts",
    "taup_traces_from_R_omega_p",
]


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GMMParams:
    """Parameters of a 2-component Gaussian mixture on ``(Δλ, Δμ, Δρ)``.

    Args:
        p_soft: Mixing weight of the soft component (``0 ≤ p_soft ≤ 1``).
        hard_mean: Hard-component means ``(Δλ, Δμ, Δρ)``.
        hard_std: Hard-component standard deviations ``(Δλ, Δμ, Δρ)``.
        soft_mean: Soft-component means.
        soft_std: Soft-component standard deviations.
    """

    p_soft: float
    hard_mean: tuple[float, float, float]
    hard_std: tuple[float, float, float]
    soft_mean: tuple[float, float, float]
    soft_std: tuple[float, float, float]


@dataclass
class VoxelConfig:
    """Voxel-lattice configuration."""

    M: int
    N_z: int
    a_km: float
    defaults: GMMParams
    overrides: dict[int, GMMParams]


@dataclass
class LayerSpec:
    """One layer of the oceanic-crust stack."""

    name: str
    alpha: float
    beta: float
    rho: float
    thickness: float
    Q_alpha: float
    Q_beta: float


@dataclass
class ScaleDemoConfig:
    """Top-level configuration for the heterogeneous τ-p demo."""

    random_seed: int
    layers: list[LayerSpec]
    voxels: VoxelConfig
    nw: int
    T_record: float
    damping: float
    p_min: float
    p_max: float
    n_p: int
    block_gmres_rtol: float
    block_gmres_max_iter: int
    output_dir: Path
    npz_name: str
    fig_wiggle_total: str
    fig_wiggle_reference: str
    fig_r_omega_p: str
    raw_yaml: str = field(default="", repr=False)

    @property
    def n_layers(self) -> int:
        """Total number of layers (water + elastic + mantle)."""
        return len(self.layers)

    @property
    def elastic_iface_range(self) -> tuple[int, int]:
        """Inclusive ``(first, last)`` interface index for scatterers.

        Scatterers sit at interfaces ``1..n_layers - 2`` — one per
        finite elastic layer, excluding the water-top and mantle
        half-space.
        """
        return 1, self.n_layers - 2


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


def _parse_gmm(block: dict[str, Any], where: str) -> GMMParams:
    """Parse a GMM block into :class:`GMMParams`.

    Args:
        block: Mapping with keys ``p_soft``, ``hard``, ``soft``.
        where: Descriptive location for error messages.

    Returns:
        Populated :class:`GMMParams`.
    """
    required = ("p_soft", "hard", "soft")
    for key in required:
        if key not in block:
            msg = f"{where}: missing required GMM key '{key}'"
            raise ValueError(msg)

    p_soft = float(block["p_soft"])
    if not (0.0 <= p_soft <= 1.0):
        msg = f"{where}: p_soft={p_soft} must be in [0, 1]"
        raise ValueError(msg)

    def _triple(src: dict[str, Any], field_name: str) -> tuple[float, float, float]:
        if field_name not in src:
            msg = f"{where}.{field_name}: missing"
            raise ValueError(msg)
        sub = src[field_name]
        out = []
        for key in ("Dlambda", "Dmu", "Drho"):
            if key not in sub:
                msg = f"{where}.{field_name}.{key}: missing"
                raise ValueError(msg)
            val = float(sub[key])
            out.append(val)
        return (out[0], out[1], out[2])

    def _triple_std(src: dict[str, Any], field_name: str) -> tuple[float, float, float]:
        if field_name not in src:
            msg = f"{where}.{field_name}: missing"
            raise ValueError(msg)
        sub = src[field_name]
        out = []
        for key in ("Dlambda", "Dmu", "Drho"):
            val = float(sub[key])
            if val < 0.0:
                msg = f"{where}.{field_name}.{key}.std={val} must be non-negative"
                raise ValueError(msg)
            out.append(val)
        return (out[0], out[1], out[2])

    hard = block["hard"]
    soft = block["soft"]
    return GMMParams(
        p_soft=p_soft,
        hard_mean=_triple({"hard": {k: hard[k]["mean"] for k in hard}}, "hard"),
        hard_std=_triple_std({"hard": {k: hard[k]["std"] for k in hard}}, "hard"),
        soft_mean=_triple({"soft": {k: soft[k]["mean"] for k in soft}}, "soft"),
        soft_std=_triple_std({"soft": {k: soft[k]["std"] for k in soft}}, "soft"),
    )


def _merge_override(base: GMMParams, patch: dict[str, Any], where: str) -> GMMParams:
    """Apply a partial override to a base GMMParams.

    The ``patch`` dict may contain any subset of
    ``{p_soft, hard.Dlambda.mean, hard.Dlambda.std, ...}`` etc.  Missing
    keys fall through to the base.

    Args:
        base: Base GMM parameters.
        patch: Partial override dict from the YAML.
        where: Descriptive location for error messages.

    Returns:
        New :class:`GMMParams` with the patch applied.
    """
    p_soft = base.p_soft
    if "p_soft" in patch:
        p_soft = float(patch["p_soft"])
        if not (0.0 <= p_soft <= 1.0):
            msg = f"{where}.p_soft={p_soft} must be in [0, 1]"
            raise ValueError(msg)

    def _merge_triple(
        base_triple: tuple[float, float, float],
        patch_sub: dict[str, Any] | None,
        key: str,
        *,
        nonneg: bool,
    ) -> tuple[float, float, float]:
        if patch_sub is None:
            return base_triple
        out = list(base_triple)
        idx_by_name = {"Dlambda": 0, "Dmu": 1, "Drho": 2}
        for name, idx in idx_by_name.items():
            if name in patch_sub and key in patch_sub[name]:
                val = float(patch_sub[name][key])
                if nonneg and val < 0.0:
                    msg = f"{where}.{name}.{key}={val} must be non-negative"
                    raise ValueError(msg)
                out[idx] = val
        return (out[0], out[1], out[2])

    hard_patch = patch.get("hard")
    soft_patch = patch.get("soft")
    return GMMParams(
        p_soft=p_soft,
        hard_mean=_merge_triple(base.hard_mean, hard_patch, "mean", nonneg=False),
        hard_std=_merge_triple(base.hard_std, hard_patch, "std", nonneg=True),
        soft_mean=_merge_triple(base.soft_mean, soft_patch, "mean", nonneg=False),
        soft_std=_merge_triple(base.soft_std, soft_patch, "std", nonneg=True),
    )


def load_scale_demo_config(path: str | Path) -> ScaleDemoConfig:
    """Load and validate a YAML configuration file.

    Args:
        path: Path to the YAML config.

    Returns:
        Populated :class:`ScaleDemoConfig`.

    Raises:
        ValueError: On any structural or value-range error, with an
            actionable message listing the offending key, value and
            expected range.
        FileNotFoundError: If the YAML file does not exist.
    """
    cfg_path = Path(path)
    if not cfg_path.is_file():
        msg = (
            f"Config file not found: {cfg_path}. "
            "Provide an existing path to a YAML configuration file."
        )
        raise FileNotFoundError(msg)
    raw_text = cfg_path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw_text)
    if not isinstance(data, dict):
        msg = f"{cfg_path}: top-level YAML node must be a mapping, got {type(data)}"
        raise ValueError(msg)

    if "random_seed" not in data:
        msg = f"{cfg_path}: missing 'random_seed'"
        raise ValueError(msg)
    random_seed = int(data["random_seed"])

    # --- layers ---
    model_block = data.get("model")
    if not isinstance(model_block, dict) or "layers" not in model_block:
        msg = f"{cfg_path}: 'model.layers' is required and must be a list"
        raise ValueError(msg)
    layer_list = model_block["layers"]
    if not isinstance(layer_list, list) or len(layer_list) < 3:
        msg = (
            f"{cfg_path}: 'model.layers' must be a list of at least 3 entries "
            "(water + >=1 elastic + mantle half-space)"
        )
        raise ValueError(msg)
    layers: list[LayerSpec] = []
    for i, entry in enumerate(layer_list):
        if not isinstance(entry, dict):
            msg = f"{cfg_path}: layers[{i}] must be a mapping"
            raise ValueError(msg)
        required = ("name", "alpha", "beta", "rho", "thickness", "Q_alpha", "Q_beta")
        for key in required:
            if key not in entry:
                msg = f"{cfg_path}: layers[{i}] missing '{key}'"
                raise ValueError(msg)
        thickness = float(entry["thickness"])
        is_last = i == len(layer_list) - 1
        if not is_last and not np.isfinite(thickness):
            msg = (
                f"{cfg_path}: layers[{i}] ('{entry['name']}') thickness={thickness} "
                "must be finite for any layer before the last. Only the final "
                "half-space may have thickness=.inf."
            )
            raise ValueError(msg)
        if thickness <= 0.0 and not is_last:
            msg = (
                f"{cfg_path}: layers[{i}] thickness={thickness} must be positive "
                "for any layer before the last."
            )
            raise ValueError(msg)
        layers.append(
            LayerSpec(
                name=str(entry["name"]),
                alpha=float(entry["alpha"]),
                beta=float(entry["beta"]),
                rho=float(entry["rho"]),
                thickness=thickness,
                Q_alpha=float(entry["Q_alpha"]),
                Q_beta=float(entry["Q_beta"]),
            )
        )

    # --- voxels ---
    voxels_block = data.get("voxels")
    if not isinstance(voxels_block, dict):
        msg = f"{cfg_path}: 'voxels' is required and must be a mapping"
        raise ValueError(msg)
    for key in ("M", "N_z", "a_km", "defaults"):
        if key not in voxels_block:
            msg = f"{cfg_path}: voxels.{key} is required"
            raise ValueError(msg)
    M_vox = int(voxels_block["M"])
    N_z_vox = int(voxels_block["N_z"])
    a_km = float(voxels_block["a_km"])
    if M_vox < 1 or N_z_vox < 1 or a_km <= 0.0:
        msg = (
            f"{cfg_path}: voxels must have M>=1, N_z>=1, a_km>0; "
            f"got M={M_vox}, N_z={N_z_vox}, a_km={a_km}"
        )
        raise ValueError(msg)
    defaults = _parse_gmm(voxels_block["defaults"], f"{cfg_path}:voxels.defaults")
    raw_overrides = voxels_block.get("overrides") or {}
    if not isinstance(raw_overrides, dict):
        msg = f"{cfg_path}: voxels.overrides must be a mapping if provided"
        raise ValueError(msg)
    n_layers = len(layers)
    iface_lo, iface_hi = 1, n_layers - 2
    overrides: dict[int, GMMParams] = {}
    for raw_key, patch in raw_overrides.items():
        key_int = int(raw_key)
        if not (iface_lo <= key_int <= iface_hi):
            msg = (
                f"{cfg_path}: voxels.overrides key {key_int} is out of range "
                f"[{iface_lo}, {iface_hi}]. Scatterer interfaces must sit "
                "strictly between the water layer and the mantle half-space."
            )
            raise ValueError(msg)
        if not isinstance(patch, dict):
            msg = f"{cfg_path}: voxels.overrides[{key_int}] must be a mapping"
            raise ValueError(msg)
        overrides[key_int] = _merge_override(
            defaults, patch, f"{cfg_path}:voxels.overrides[{key_int}]"
        )

    voxels = VoxelConfig(
        M=M_vox,
        N_z=N_z_vox,
        a_km=a_km,
        defaults=defaults,
        overrides=overrides,
    )

    # --- frequency / slowness / gmres ---
    freq_block = data.get("frequency") or {}
    nw = int(freq_block.get("nw", 256))
    T_record = float(freq_block.get("T_record", 32.0))
    damping = float(freq_block.get("damping", 0.1))
    if nw < 1 or T_record <= 0.0:
        msg = f"{cfg_path}: frequency.nw>=1 and T_record>0 required"
        raise ValueError(msg)

    slow_block = data.get("slowness") or {}
    p_min = float(slow_block.get("p_min", 0.05))
    p_max = float(slow_block.get("p_max", 0.60))
    n_p = int(slow_block.get("n_p", 48))
    if p_min < 0.0 or p_max <= p_min or n_p < 2:
        msg = (
            f"{cfg_path}: require slowness.p_min>=0, p_max>p_min, n_p>=2; "
            f"got p_min={p_min}, p_max={p_max}, n_p={n_p}"
        )
        raise ValueError(msg)

    gmres_block = data.get("block_gmres") or {}
    rtol = float(gmres_block.get("rtol", 1.0e-8))
    max_iter = int(gmres_block.get("max_iter", 30))

    # --- output ---
    out_block = data.get("output") or {}
    output_dir = Path(out_block.get("dir", "GlobalMatrix/output/taup_scale_demo"))
    npz_name = str(out_block.get("npz_name", "taup_scale_demo.npz"))
    figures = out_block.get("figures") or {}
    fig_wiggle_total = str(figures.get("wiggle_total", "taup_wiggle_gather.pdf"))
    fig_wiggle_reference = str(
        figures.get("wiggle_reference", "taup_reference_wiggle_gather.pdf")
    )
    fig_r_omega_p = str(figures.get("r_omega_p", "r_omega_p_heatmap.pdf"))

    return ScaleDemoConfig(
        random_seed=random_seed,
        layers=layers,
        voxels=voxels,
        nw=nw,
        T_record=T_record,
        damping=damping,
        p_min=p_min,
        p_max=p_max,
        n_p=n_p,
        block_gmres_rtol=rtol,
        block_gmres_max_iter=max_iter,
        output_dir=output_dir,
        npz_name=npz_name,
        fig_wiggle_total=fig_wiggle_total,
        fig_wiggle_reference=fig_wiggle_reference,
        fig_r_omega_p=fig_r_omega_p,
        raw_yaml=raw_text,
    )


# ---------------------------------------------------------------------------
# Model assembly
# ---------------------------------------------------------------------------


def build_oceanic_crust_model(config: ScaleDemoConfig) -> LayerModel:
    """Build the ``LayerModel`` from the config's layer stack.

    Args:
        config: Parsed scale-demo configuration.

    Returns:
        A :class:`LayerModel` with ``n_layers = len(config.layers)``.
    """
    alpha = np.asarray([layer.alpha for layer in config.layers], dtype=np.float64)
    beta = np.asarray([layer.beta for layer in config.layers], dtype=np.float64)
    rho = np.asarray([layer.rho for layer in config.layers], dtype=np.float64)
    thickness = np.asarray(
        [layer.thickness for layer in config.layers], dtype=np.float64
    )
    Q_alpha = np.asarray([layer.Q_alpha for layer in config.layers], dtype=np.float64)
    Q_beta = np.asarray([layer.Q_beta for layer in config.layers], dtype=np.float64)
    return LayerModel(
        alpha=alpha,
        beta=beta,
        rho=rho,
        thickness=thickness,
        Q_alpha=Q_alpha,
        Q_beta=Q_beta,
    )


# ---------------------------------------------------------------------------
# Voxel sampling
# ---------------------------------------------------------------------------


def _layer_lame(layer: LayerSpec) -> tuple[float, float, float]:
    """Compute Lamé parameters ``(λ, μ, ρ)`` from a layer's elastic properties.

    Uses the elastic relations
    ``μ = ρ β²`` and ``λ = ρ (α² − 2 β²)``.

    Args:
        layer: One layer of the model.

    Returns:
        Tuple ``(λ, μ, ρ)`` in GPa, GPa, g/cm³.
    """
    rho = float(layer.rho)
    mu = rho * float(layer.beta) ** 2
    lam = rho * (float(layer.alpha) ** 2 - 2.0 * float(layer.beta) ** 2)
    return lam, mu, rho


def _sample_one(
    rng: np.random.Generator,
    gmm: GMMParams,
    layer: LayerSpec,
) -> MaterialContrast:
    """Draw a single 2-component Gaussian-mixture *fractional* contrast sample.

    The GMM ``mean`` and ``std`` triples are interpreted as fractional
    perturbations of the layer's background ``(λ, μ, ρ)``.  This makes
    a single GMM physically sensible across layers whose Lamé
    parameters span several orders of magnitude (e.g. soft sediments
    with μ ≈ 0.3 GPa vs mantle with μ ≈ 67 GPa).

    Args:
        rng: Numpy random generator.
        gmm: Mixture parameters (interpreted as fractions).
        layer: Layer the sample is for; supplies the background scale.

    Returns:
        A :class:`MaterialContrast` whose magnitudes are
        ``frac · (λ_layer, μ_layer, ρ_layer)``.
    """
    lam, mu, rho = _layer_lame(layer)
    if rng.uniform() < gmm.p_soft:
        mean, std = gmm.soft_mean, gmm.soft_std
    else:
        mean, std = gmm.hard_mean, gmm.hard_std
    return MaterialContrast(
        Dlambda=float(rng.normal(mean[0], std[0])) * lam,
        Dmu=float(rng.normal(mean[1], std[1])) * mu,
        Drho=float(rng.normal(mean[2], std[2])) * rho,
    )


def sample_voxel_contrasts(
    config: ScaleDemoConfig,
    rng: np.random.Generator,
) -> dict[int, list[MaterialContrast]]:
    """Sample per-voxel :class:`MaterialContrast` lists for each elastic iface.

    The returned dict maps interface index (``1..n_layers - 2``) to a
    list of ``M*M*N_z`` fractional-contrast samples, in the canonical
    :func:`cluster_from_slab` ordering (``iz`` outer, ``ix``, ``iy``
    innermost).

    GMM ``mean`` / ``std`` values in the YAML are interpreted as
    fractional perturbations of the layer's background
    ``(λ_layer, μ_layer, ρ_layer)``.  This is the only physically
    sensible interpretation when the model spans 100× ranges in
    background ``μ``.

    For each interface, the GMM parameters are taken from
    ``config.voxels.overrides[iface]`` if present, otherwise from
    ``config.voxels.defaults``.

    Args:
        config: Parsed scale-demo configuration.
        rng: Numpy random generator (seeded by the caller).

    Returns:
        Dict mapping elastic iface index to a list of length
        ``M*M*N_z`` of :class:`MaterialContrast`.
    """
    iface_lo, iface_hi = config.elastic_iface_range
    M_vox = config.voxels.M
    N_z_vox = config.voxels.N_z
    nC = M_vox * M_vox * N_z_vox

    result: dict[int, list[MaterialContrast]] = {}
    for iface in range(iface_lo, iface_hi + 1):
        gmm = config.voxels.overrides.get(iface, config.voxels.defaults)
        layer = config.layers[iface]
        contrasts = [_sample_one(rng, gmm, layer) for _ in range(nC)]
        result[iface] = contrasts
    return result


# ---------------------------------------------------------------------------
# Per-layer inner solves
# ---------------------------------------------------------------------------


def _worker_init(extra_paths: list[str]) -> None:
    """Initializer run once per worker process on spawn.

    Adds project paths to ``sys.path`` so that
    ``GlobalMatrix.taup_scale_demo`` and its siblings are importable
    inside the worker, and clamps BLAS/OMP thread pools to a single
    thread to prevent CPU oversubscription when N workers each spawn
    BLAS thread pools sized to the full core count.

    Must be module-level (not a closure) so it is picklable under the
    ``spawn`` start method.
    """
    for key in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(key, "1")
    for p in extra_paths:
        if p and p not in sys.path:
            sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Per-(iface, freq) disk cache
# ---------------------------------------------------------------------------
#
# At fixed ω the Helmholtz operator is fixed: all ``n_p_in`` incident
# slownesses become extra RHS columns of one block GMRES.  The natural
# unit of work is therefore one (iface, freq) pair, each producing a
# full slowness-coupling block ``T(p_out, p_in, 9, 9)`` for that
# (layer, frequency).  We persist each block as a small .npz under
# ``cache_dir/iface_{j:02d}/freq_{f:04d}.npz`` so the inner phase is
# crash-resumable and decoupled from the outer Riccati sweep.
#
# Cache invalidation is automatic: a SHA-256 cache key over every
# physically-relevant input (lattice, reference medium, ω, slowness
# grids, per-cube contrasts, GMRES tolerances, k_par convention) is
# stored in the .npz; on a hit the key is checked and the file is
# ignored if anything drifted.


def _make_cache_key(
    *,
    iface: int,
    freq_idx: int,
    omega: complex,
    alpha: float,
    beta: float,
    rho: float,
    M_vox: int,
    N_z_vox: int,
    a_km: float,
    slowness_in_2d: NDArray[np.floating],
    slowness_out_2d: NDArray[np.floating],
    contrasts: list[MaterialContrast],
    rtol: float,
    max_iter: int,
) -> str:
    """SHA-256 hex digest over every input that affects ``T_block``.

    Used as the cache validation key.  Any change in lattice, reference
    medium, ω, 2D slowness grids, per-cube contrasts, GMRES tolerances,
    or slowness convention triggers a cache miss.

    Args:
        iface: Interface index.
        freq_idx: Frequency index.
        omega: Single complex angular frequency at this slot.
        alpha, beta, rho: Reference-medium elastic parameters.
        M_vox, N_z_vox, a_km: Voxel-lattice parameters.
        slowness_in_2d: Incident 2D slowness grid (s/km), shape
            ``(n_p_in, 2)`` with columns ``(p_x, p_y)``.
        slowness_out_2d: Outgoing 2D slowness grid (s/km), shape
            ``(n_p_out, 2)`` with columns ``(p_x, p_y)``.
        contrasts: Per-cube material contrasts (length ``M*M*N_z``).
        rtol, max_iter: Block GMRES tolerances.

    Returns:
        64-character SHA-256 hex digest.
    """
    h = hashlib.sha256()
    h.update(
        json.dumps(
            {
                "iface": int(iface),
                "freq_idx": int(freq_idx),
                "omega_real": float(omega.real),
                "omega_imag": float(omega.imag),
                "alpha": float(alpha),
                "beta": float(beta),
                "rho": float(rho),
                "M_vox": int(M_vox),
                "N_z_vox": int(N_z_vox),
                "a_km": float(a_km),
                "rtol": float(rtol),
                "max_iter": int(max_iter),
                "k_par_convention": (
                    "kx=omega*p_x, ky=omega*p_y, "
                    "kz=sqrt((omega/beta)**2 - kx**2 - ky**2)"
                ),
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    p_in_arr = np.ascontiguousarray(slowness_in_2d, dtype=np.float64)
    p_out_arr = np.ascontiguousarray(slowness_out_2d, dtype=np.float64)
    h.update(b"slowness_in_2d:")
    h.update(p_in_arr.tobytes())
    h.update(b"slowness_out_2d:")
    h.update(p_out_arr.tobytes())
    contrasts_arr = np.array(
        [(c.Dlambda, c.Dmu, c.Drho) for c in contrasts], dtype=np.float64
    )
    h.update(b"contrasts:")
    h.update(contrasts_arr.tobytes())
    return h.hexdigest()


def _atomic_save_npz(path: Path, **arrays: Any) -> None:
    """Save ``.npz`` atomically via ``write-then-rename``.

    Writes to ``{path}.tmp`` first, then renames to ``{path}``.  Ensures
    that a concurrent reader (or a crash mid-write) never observes a
    half-written cache file.

    ``np.savez`` auto-appends ``.npz`` to path-like arguments that don't
    already end in ``.npz``, so we pass an open file handle instead to
    force it to write to the exact temp path we chose.

    Args:
        path: Final destination path.
        **arrays: Keyword arguments forwarded to ``np.savez``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp_path, path)


def _cache_path_for(cache_dir: Path, iface: int, freq_idx: int) -> Path:
    """Return the canonical cache path for ``(iface, freq_idx)``."""
    return cache_dir / f"iface_{iface:02d}" / f"freq_{freq_idx:04d}.npz"


def _load_cached_layer_freq(
    cache_dir: Path,
    iface: int,
    freq_idx: int,
    cache_key: str,
) -> NDArray[np.complexfloating] | None:
    """Try to load a cached ``(n_p_out, n_p_in, 9, 9)`` block for ``(iface, freq_idx)``.

    Returns ``None`` on a cache miss (file absent, key mismatch, or
    corrupt file — corrupt files are silently ignored to be resumable
    after a crash).

    Args:
        cache_dir: Root cache directory.
        iface: Interface index.
        freq_idx: Frequency index.
        cache_key: Expected SHA-256 cache key.

    Returns:
        Cached ``T_block`` array, or ``None`` on miss.
    """
    path = _cache_path_for(cache_dir, iface, freq_idx)
    if not path.is_file():
        return None
    try:
        with np.load(path) as data:
            if str(data["cache_key"]) != cache_key:
                return None
            return np.asarray(data["T_block"], dtype=complex)
    except (OSError, KeyError, ValueError):
        return None


def _kpar_array_for_omega(
    omega: complex,
    beta: float,
    slowness_2d: NDArray[np.floating],
) -> NDArray[np.complexfloating]:
    """Build ``k_par[k] = (ω p_x, ω p_y, sqrt((ω/β)² − k_par²))`` at one ω.

    Each row of ``slowness_2d`` is an arbitrary horizontal slowness pair
    ``(p_x, p_y)`` in s/km.  The vertical wavenumber is derived from the
    elastic dispersion of the local reference S-wave speed via
    ``kz = sqrt((ω/β)² − (ω p_x)² − (ω p_y)²)``.  The complex square
    root preserves the post-critical (evanescent) branch automatically:
    when ``|k_par| > ω/β``, ``kz`` becomes purely imaginary with the
    physically correct decaying sign for the propagation half-space.

    The 2D-aware API lifts the ``p_y = 0`` restriction of the original
    helper.  Single-azimuth τ-p along x-axis is the special case
    ``slowness_2d = np.column_stack([p_values, np.zeros_like(p_values)])``;
    full azimuthal coverage uses any other 2D grid (e.g. tensor-product
    ``(p_x, p_y)`` or the FFT-native lattice).

    Args:
        omega: Single complex angular frequency.
        beta: Reference S-wave speed (km/s).
        slowness_2d: Horizontal slowness grid in s/km, shape ``(n_p, 2)``,
            with columns ``(p_x, p_y)``.

    Returns:
        Per-slowness wave-vector stack, shape ``(n_p, 3)``, complex,
        with columns ``(k_x, k_y, k_z)``.
    """
    p_arr = np.asarray(slowness_2d, dtype=np.float64)
    if p_arr.ndim != 2 or p_arr.shape[1] != 2:
        msg = (
            f"slowness_2d has shape {p_arr.shape}, expected (n_p, 2) "
            "with columns (p_x, p_y)."
        )
        raise ValueError(msg)
    om = complex(omega)
    kx = om * p_arr[:, 0]
    ky = om * p_arr[:, 1]
    kz_sq = (om / beta) ** 2 - kx**2 - ky**2
    kz = np.sqrt(kz_sq)  # complex sqrt — branch is correct for evanescent
    out = np.zeros((p_arr.shape[0], 3), dtype=complex)
    out[:, 0] = kx
    out[:, 1] = ky
    out[:, 2] = kz
    return out


def _solve_layer_freq_het_worker(
    iface: int,
    freq_idx: int,
    omega: complex,
    layer_name: str,
    alpha: float,
    beta: float,
    rho: float,
    M_vox: int,
    N_z_vox: int,
    a_km: float,
    slowness_in_2d: NDArray[np.floating],
    slowness_out_2d: NDArray[np.floating],
    contrasts: list[MaterialContrast],
    rtol: float,
    max_iter: int,
    cache_dir_str: str | None,
    force_recompute: bool,
) -> tuple[
    int,  # iface
    int,  # freq_idx
    str,  # layer_name
    NDArray[np.complexfloating],  # T_block, shape (n_p_out, n_p_in, 9, 9)
    int,  # iters
    float,  # rel_res
    bool,  # was_cached
]:
    """Per-(iface, freq) **slowness-batched** inner solve with disk caching.

    Builds the incident and outgoing ``k_par`` stacks for ``omega`` from
    the 2D slowness grids, calls
    :func:`solve_slab_foldy_lax_single_freq_het_taup` once (one block
    GMRES with ``9·n_p_in`` RHS columns), and persists the resulting
    ``(n_p_out, n_p_in, 9, 9)`` block to
    ``cache_dir/iface_{j:02d}/freq_{f:04d}.npz`` (atomic write).  On a
    cache hit with matching ``cache_key`` the solve is skipped entirely.

    Module-level (not a closure) so it is picklable under the ``spawn``
    start method.

    Args:
        iface: Interface index.
        freq_idx: Frequency index.
        omega: Single complex angular frequency at this slot.
        layer_name: Layer name (for verbose logging upstream).
        alpha, beta, rho: Reference-medium parameters.
        M_vox, N_z_vox, a_km: Voxel-lattice parameters.
        slowness_in_2d: Incident 2D slowness grid (s/km), shape
            ``(n_p_in, 2)`` with columns ``(p_x, p_y)``.
        slowness_out_2d: Outgoing 2D slowness grid (s/km), shape
            ``(n_p_out, 2)`` with columns ``(p_x, p_y)``.
        contrasts: Per-cube material contrasts.
        rtol, max_iter: Block GMRES tolerances.
        cache_dir_str: Root cache directory as a string (or ``None`` to
            disable caching for this job).  String rather than ``Path``
            so the args tuple is trivially picklable.
        force_recompute: If ``True``, ignore any existing cache hit and
            re-run the solve (the new result still overwrites the file).

    Returns:
        Tuple ``(iface, freq_idx, layer_name, T_block, iters, rel_res,
        was_cached)``.  When ``was_cached`` is ``True`` the ``iters``
        and ``rel_res`` fields are loaded from the cache file.
    """
    cache_key = _make_cache_key(
        iface=iface,
        freq_idx=freq_idx,
        omega=omega,
        alpha=alpha,
        beta=beta,
        rho=rho,
        M_vox=M_vox,
        N_z_vox=N_z_vox,
        a_km=a_km,
        slowness_in_2d=slowness_in_2d,
        slowness_out_2d=slowness_out_2d,
        contrasts=contrasts,
        rtol=rtol,
        max_iter=max_iter,
    )

    cache_dir = Path(cache_dir_str) if cache_dir_str is not None else None
    if cache_dir is not None and not force_recompute:
        cached = _load_cached_layer_freq(cache_dir, iface, freq_idx, cache_key)
        if cached is not None:
            path = _cache_path_for(cache_dir, iface, freq_idx)
            with np.load(path) as data:
                iters_cached = int(data["iters"])
                rel_res_cached = float(data["rel_res"])
            return (
                iface,
                freq_idx,
                layer_name,
                cached,
                iters_cached,
                rel_res_cached,
                True,
            )

    ref = ReferenceMedium(alpha=alpha, beta=beta, rho=rho)
    k_par_in = _kpar_array_for_omega(omega, beta, slowness_in_2d)
    k_par_out = _kpar_array_for_omega(omega, beta, slowness_out_2d)

    T_block, iters_int, rel_res_val = solve_slab_foldy_lax_single_freq_het_taup(
        M=M_vox,
        N_z=N_z_vox,
        a=a_km,
        omega=omega,
        ref=ref,
        contrasts_per_cube=contrasts,
        k_par_in=k_par_in,
        k_par_out=k_par_out,
        rtol=rtol,
        max_iter=max_iter,
    )

    if cache_dir is not None:
        path = _cache_path_for(cache_dir, iface, freq_idx)
        _atomic_save_npz(
            path,
            T_block=T_block,
            iters=np.int64(iters_int),
            rel_res=np.float64(rel_res_val),
            cache_key=np.array(cache_key),
            iface=np.int64(iface),
            freq_idx=np.int64(freq_idx),
            omega=np.complex128(omega),
            slowness_in_2d=np.asarray(slowness_in_2d, dtype=np.float64),
            slowness_out_2d=np.asarray(slowness_out_2d, dtype=np.float64),
        )

    return (
        iface,
        freq_idx,
        layer_name,
        T_block,
        int(iters_int),
        float(rel_res_val),
        False,
    )


def load_cached_tmatrices(
    cache_dir: str | Path,
    slowness_in_2d: NDArray[np.floating],
    slowness_out_2d: NDArray[np.floating],
    n_freq: int,
) -> dict[int, NDArray[np.complexfloating]]:
    """Re-load every cached ``(F, n_p_out, n_p_in, 9, 9)`` T-stack from ``cache_dir``.

    Walks ``cache_dir/iface_*/freq_*.npz``, validates each block's
    shape against ``(n_p_out, n_p_in, 9, 9)``, and assembles one
    ``(F, n_p_out, n_p_in, 9, 9)`` array per interface by stacking
    along the frequency axis.  Skips any interface for which one or
    more ``(freq_idx)`` files are missing.

    Args:
        cache_dir: Root cache directory.
        slowness_in_2d: Incident 2D slowness grid (s/km), shape
            ``(n_p_in, 2)`` with columns ``(p_x, p_y)``.
        slowness_out_2d: Outgoing 2D slowness grid (s/km), shape
            ``(n_p_out, 2)`` with columns ``(p_x, p_y)``.
        n_freq: Expected number of frequency slots in the cache.

    Returns:
        Dict mapping interface index →
        ``(F, n_p_out, n_p_in, 9, 9)`` complex array.  Interfaces with
        any missing or mismatched cache file are omitted from the dict.

    Raises:
        FileNotFoundError: If ``cache_dir`` does not exist.
    """
    cdir = Path(cache_dir)
    if not cdir.is_dir():
        msg = (
            f"Cache directory not found: {cdir}. "
            "Did you forget to run the inner solver, or pass the wrong path?"
        )
        raise FileNotFoundError(msg)

    n_p_in = int(np.asarray(slowness_in_2d).shape[0])
    n_p_out = int(np.asarray(slowness_out_2d).shape[0])
    iface_dirs = sorted(cdir.glob("iface_*"))
    out: dict[int, NDArray[np.complexfloating]] = {}
    for iface_dir in iface_dirs:
        try:
            iface = int(iface_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        T_stack = np.zeros((n_freq, n_p_out, n_p_in, 9, 9), dtype=complex)
        complete = True
        for f_idx in range(n_freq):
            path = _cache_path_for(cdir, iface, f_idx)
            if not path.is_file():
                complete = False
                break
            with np.load(path) as data:
                T_block = np.asarray(data["T_block"], dtype=complex)
                if T_block.shape != (n_p_out, n_p_in, 9, 9):
                    complete = False
                    break
                T_stack[f_idx] = T_block
        if complete:
            out[iface] = T_stack
    return out


def _solve_single_layer_het_worker(
    iface: int,
    layer_name: str,
    alpha: float,
    beta: float,
    rho: float,
    M_vox: int,
    N_z_vox: int,
    a_km: float,
    omegas: NDArray[np.complexfloating],
    contrasts: list[MaterialContrast],
    batch_sizes: list[int],
    rtol: float,
    max_iter: int,
) -> tuple[int, str, NDArray[np.complexfloating], int, NDArray[np.floating]]:
    """Per-layer block Foldy-Lax inner solve (picklable for ProcessPoolExecutor).

    Runs one or more frequency-batched calls of
    :func:`solve_slab_foldy_lax_freq_het` and concatenates the chunks
    along the frequency axis.  This is the unit of work dispatched to
    worker processes when ``n_workers > 1``.

    Args:
        iface: Interface index (for result association).
        layer_name: Layer name (for verbose logging in the parent).
        alpha, beta, rho: Reference-medium elastic parameters.
        M_vox: Voxel lattice edge (M × M in-plane).
        N_z_vox: Number of z-slices of cubes.
        a_km: Cube half-width in km.
        omegas: Complex angular frequencies, shape ``(F,)``.
        contrasts: Per-cube material contrasts (length ``M*M*N_z``).
        batch_sizes: Frequency-batch sizes that sum to ``F``.
        rtol: Block GMRES relative-residual tolerance.
        max_iter: Block GMRES maximum iterations.

    Returns:
        Tuple ``(iface, layer_name, T_freq, max_iters, res)`` where
        ``T_freq`` has shape ``(F, 9, 9)`` and ``res`` has shape ``(F,)``.
    """
    ref = ReferenceMedium(alpha=alpha, beta=beta, rho=rho)
    T_chunks: list[NDArray[np.complexfloating]] = []
    res_chunks: list[NDArray[np.floating]] = []
    max_iters = 0
    cursor = 0
    for step in batch_sizes:
        omegas_chunk = omegas[cursor : cursor + step]
        T_chunk, it_chunk, res_chunk = solve_slab_foldy_lax_freq_het(
            M=M_vox,
            N_z=N_z_vox,
            a=a_km,
            omegas=omegas_chunk,
            ref=ref,
            contrasts_per_cube=contrasts,
            rtol=rtol,
            max_iter=max_iter,
        )
        T_chunks.append(T_chunk)
        res_chunks.append(res_chunk)
        max_iters = max(max_iters, int(it_chunk))
        cursor += step
    T_freq = np.concatenate(T_chunks, axis=0)
    res = np.concatenate(res_chunks, axis=0)
    return iface, layer_name, T_freq, max_iters, res


def compute_layer_composite_tmatrices_het(
    config: ScaleDemoConfig,
    voxel_contrasts: dict[int, list[MaterialContrast]],
    omegas: NDArray[np.complexfloating],
    *,
    slowness_2d: NDArray[np.floating] | None = None,
    cache_dir: str | Path | None = None,
    force_recompute: bool = False,
    freq_batch_size: int | None = None,
    n_workers: int | None = None,
    verbose: bool = False,
) -> tuple[
    dict[int, NDArray[np.complexfloating]],
    dict[int, int],
    dict[int, NDArray[np.floating]],
]:
    """Per-layer heterogeneous block Foldy-Lax → composite T stack.

    Two operating modes, controlled by ``slowness_2d``:

    * **k-space mode** (``slowness_2d is None``, legacy): for each
      elastic interface, runs one freq-batched solve at normal incidence
      (``k_hat = ẑ``, ``wave_type = "S"``).  Returns
      ``dict[int, (F, 9, 9)]`` — the original semantics.

    * **τ-p mode** (``slowness_2d`` provided as a 2D ``(n_p, 2)`` array
      of ``(p_x, p_y)`` pairs in s/km): for each elastic interface and
      each frequency ``ω_f``, runs **one** slowness-batched solve via
      :func:`solve_slab_foldy_lax_single_freq_het_taup` with all ``n_p``
      incident 2D slownesses stacked as the RHS columns and the same
      grid as the outgoing-slowness projection target.  Returns
      ``dict[int, (F, n_p, n_p, 9, 9)]`` — the full 2D slowness
      coupling block at every (iface, ω).  Single-azimuth τ-p along
      x-axis is the special case where every row has ``p_y = 0``;
      arbitrary 2D grids (tensor-product, FFT-native, off-axis) are
      handled identically.  At full scale this is
      ``18 × 256 = 4608`` independent jobs; the work is dispatched to
      a :class:`ProcessPoolExecutor` and each job's result is persisted
      to ``cache_dir`` so the inner phase is crash-resumable and can be
      re-loaded by :func:`load_cached_tmatrices` after the fact.

    Layers whose contrasts are all zero (e.g. from a user override
    forcing all Δλ=Δμ=Δρ=0) are skipped in both modes.

    When ``freq_batch_size`` is set in k-space mode, the frequency grid
    is split into chunks and the solver is called once per chunk.  In
    τ-p mode each (iface, freq) job is one block GMRES with
    ``9·n_p`` RHS columns; parallelism is across (iface, freq) and
    each worker clamps BLAS threads to 1 to avoid oversubscription.

    Args:
        config: Scale-demo configuration.
        voxel_contrasts: Dict mapping iface → per-cube contrasts.
        omegas: Complex angular frequencies, shape ``(F,)``.
        slowness_2d: Optional 2D horizontal slowness grid (s/km) of
            shape ``(n_p, 2)`` with columns ``(p_x, p_y)``.  When
            ``None``, the legacy normal-incidence path is used.  When
            provided, the output T-stack has shape
            ``(F, n_p, n_p, 9, 9)`` per interface — the full 2D
            slowness coupling block at every ω.
        cache_dir: Directory to persist per-(iface, freq) results in.
            Required for τ-p mode (highly recommended for crash
            resumability).  Ignored in k-space mode.
        force_recompute: When ``True``, ignore any existing cache hits
            and re-run every job.  The fresh result still overwrites
            the file.  Ignored in k-space mode.
        freq_batch_size: Optional max frequencies per block GMRES call.
            k-space mode only.
        n_workers: Optional number of worker processes for parallel
            execution.  ``None`` or ``1`` runs jobs sequentially.
            Values ``≥ 2`` use :class:`ProcessPoolExecutor` with the
            ``spawn`` start method; BLAS thread pools are clamped to
            one thread per worker.
        verbose: If True, print per-job iteration/residual summary.

    Returns:
        Tuple ``(tmatrices, iters, rel_res)`` where:

        * In k-space mode: ``tmatrices[j]`` has shape ``(F, 9, 9)``,
          ``iters[j]`` is the max iterations across frequency chunks,
          and ``rel_res[j]`` has shape ``(F,)``.
        * In τ-p mode: ``tmatrices[j]`` has shape
          ``(F, n_p, n_p, 9, 9)``, ``iters[j]`` is the max iterations
          across all ``(freq_idx)`` jobs for that interface, and
          ``rel_res[j]`` has shape ``(F,)``.
    """
    if slowness_2d is not None:
        return _compute_layer_composite_tmatrices_het_taup(
            config=config,
            voxel_contrasts=voxel_contrasts,
            omegas=omegas,
            slowness_2d=slowness_2d,
            cache_dir=cache_dir,
            force_recompute=force_recompute,
            n_workers=n_workers,
            verbose=verbose,
        )

    M_vox = config.voxels.M
    N_z_vox = config.voxels.N_z
    a_km = config.voxels.a_km
    n_freq = int(omegas.shape[0])
    if freq_batch_size is None or freq_batch_size <= 0 or freq_batch_size >= n_freq:
        batch_sizes = [n_freq]
    else:
        batch_sizes = []
        remaining = n_freq
        while remaining > 0:
            step = min(freq_batch_size, remaining)
            batch_sizes.append(step)
            remaining -= step

    tmatrices: dict[int, NDArray[np.complexfloating]] = {}
    iters: dict[int, int] = {}
    rel_res: dict[int, NDArray[np.floating]] = {}

    # Pre-filter: build the list of layers that actually need solving.
    # Layers with all-zero contrasts are a user escape hatch to skip
    # a particular interface without touching the solver.
    pending: list[tuple[int, LayerSpec, list[MaterialContrast]]] = []
    for iface, contrasts in voxel_contrasts.items():
        if all(c.Dlambda == 0.0 and c.Dmu == 0.0 and c.Drho == 0.0 for c in contrasts):
            if verbose:
                print(f"  iface {iface}: all-zero contrasts; skipping")
            continue
        pending.append((iface, config.layers[iface], contrasts))

    if not pending:
        return tmatrices, iters, rel_res

    use_parallel = n_workers is not None and n_workers >= 2 and len(pending) >= 2

    if not use_parallel:
        for iface, layer, contrasts in pending:
            _, _, T_freq, max_iters, res = _solve_single_layer_het_worker(
                iface=iface,
                layer_name=layer.name,
                alpha=layer.alpha,
                beta=layer.beta,
                rho=layer.rho,
                M_vox=M_vox,
                N_z_vox=N_z_vox,
                a_km=a_km,
                omegas=omegas,
                contrasts=contrasts,
                batch_sizes=batch_sizes,
                rtol=config.block_gmres_rtol,
                max_iter=config.block_gmres_max_iter,
            )
            tmatrices[iface] = T_freq
            iters[iface] = max_iters
            rel_res[iface] = res
            if verbose:
                max_res = float(np.max(res))
                print(
                    f"  iface {iface:2d} ({layer.name:>10s}): "
                    f"iters≤{max_iters:2d}, max rel res={max_res:.2e}"
                )
        return tmatrices, iters, rel_res

    # Layer-parallel path: spawn one worker per CPU (up to n_workers),
    # feed each independent layer as a separate task.  Spawn context
    # avoids fork-related issues with macOS + numpy + scipy.  The
    # initializer sets up sys.path and clamps BLAS thread pools to 1.
    n_effective = min(int(n_workers), len(pending))  # type: ignore[arg-type]
    project_root = str(Path(__file__).resolve().parents[1])
    sibling_root = str(Path(project_root).parent / "MultipleScatteringCalculations")
    extra_paths = [project_root, sibling_root]

    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=n_effective,
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(extra_paths,),
    ) as executor:
        futures = {
            executor.submit(
                _solve_single_layer_het_worker,
                iface,
                layer.name,
                layer.alpha,
                layer.beta,
                layer.rho,
                M_vox,
                N_z_vox,
                a_km,
                omegas,
                contrasts,
                batch_sizes,
                config.block_gmres_rtol,
                config.block_gmres_max_iter,
            ): iface
            for iface, layer, contrasts in pending
        }
        for fut in as_completed(futures):
            iface, layer_name, T_freq, max_iters, res = fut.result()
            tmatrices[iface] = T_freq
            iters[iface] = max_iters
            rel_res[iface] = res
            if verbose:
                max_res = float(np.max(res))
                print(
                    f"  iface {iface:2d} ({layer_name:>10s}): "
                    f"iters≤{max_iters:2d}, max rel res={max_res:.2e}"
                )

    # Stable dict ordering by iface so downstream code that iterates
    # sees layers in canonical order.
    tmatrices = {k: tmatrices[k] for k in sorted(tmatrices)}
    iters = {k: iters[k] for k in sorted(iters)}
    rel_res = {k: rel_res[k] for k in sorted(rel_res)}
    return tmatrices, iters, rel_res


def _compute_layer_composite_tmatrices_het_taup(
    *,
    config: ScaleDemoConfig,
    voxel_contrasts: dict[int, list[MaterialContrast]],
    omegas: NDArray[np.complexfloating],
    slowness_2d: NDArray[np.floating],
    cache_dir: str | Path | None,
    force_recompute: bool,
    n_workers: int | None,
    verbose: bool,
) -> tuple[
    dict[int, NDArray[np.complexfloating]],
    dict[int, int],
    dict[int, NDArray[np.floating]],
]:
    """τ-p-mode body of :func:`compute_layer_composite_tmatrices_het`.

    Slowness-batched architecture: at fixed ω the Helmholtz operator is
    fixed, so all ``n_p`` incident 2D slownesses become extra RHS
    columns of one block GMRES.  The unit of work is therefore one
    ``(iface, freq_idx)`` pair, and one call produces the full
    ``(n_p, n_p, 9, 9)`` slowness-coupling block for that
    layer-frequency.

    The ``slowness_2d`` grid is used as both the incident and the
    outgoing slowness grid (n_p_in = n_p_out = n_p): the slowness
    coupling matrix is square and the multi-slowness Riccati outer
    sweep consumes the same grid on both legs.  Each row is an
    arbitrary horizontal slowness pair ``(p_x, p_y)``; single-azimuth
    τ-p along x-axis is the special case ``p_y = 0``.
    """
    M_vox = config.voxels.M
    N_z_vox = config.voxels.N_z
    a_km = config.voxels.a_km
    n_freq = int(omegas.shape[0])
    slowness_arr = np.asarray(slowness_2d, dtype=np.float64)
    if slowness_arr.ndim != 2 or slowness_arr.shape[1] != 2:
        msg = (
            f"slowness_2d has shape {slowness_arr.shape}, "
            "expected (n_p, 2) with columns (p_x, p_y)."
        )
        raise ValueError(msg)
    n_p = int(slowness_arr.shape[0])
    slowness_in_2d = slowness_arr
    slowness_out_2d = slowness_arr  # square coupling on the same 2D grid

    cache_dir_path = Path(cache_dir) if cache_dir is not None else None
    if cache_dir_path is not None:
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        # Drop a human-readable manifest so future debugging can pin
        # down which run produced these files.
        meta = {
            "M_vox": M_vox,
            "N_z_vox": N_z_vox,
            "a_km": a_km,
            "n_freq": n_freq,
            "n_p_in": n_p,
            "n_p_out": n_p,
            "slowness_2d": [[float(p) for p in row] for row in slowness_in_2d],
            "omegas_real": [float(o.real) for o in omegas],
            "omegas_imag": [float(o.imag) for o in omegas],
            "rtol": float(config.block_gmres_rtol),
            "max_iter": int(config.block_gmres_max_iter),
            "k_par_convention": (
                "kx=omega*p_x, ky=omega*p_y, kz=sqrt((omega/beta)**2 - kx**2 - ky**2)"
            ),
            "random_seed": int(config.random_seed),
            "architecture": "slowness-batched per (iface, freq)",
        }
        (cache_dir_path / "cache_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

    # Pre-filter: build the list of layers that actually need solving.
    pending_layers: list[tuple[int, LayerSpec, list[MaterialContrast]]] = []
    for iface, contrasts in voxel_contrasts.items():
        if all(c.Dlambda == 0.0 and c.Dmu == 0.0 and c.Drho == 0.0 for c in contrasts):
            if verbose:
                print(f"  iface {iface}: all-zero contrasts; skipping")
            continue
        pending_layers.append((iface, config.layers[iface], contrasts))

    tmatrices: dict[int, NDArray[np.complexfloating]] = {}
    iters: dict[int, int] = {}
    rel_res: dict[int, NDArray[np.floating]] = {}

    if not pending_layers:
        return tmatrices, iters, rel_res

    # Allocate per-iface (F, n_p_out, n_p_in, 9, 9) buffers up front so
    # workers can stream their results into them via (iface, freq_idx)
    # routing as completions arrive.
    for iface, _, _ in pending_layers:
        tmatrices[iface] = np.zeros((n_freq, n_p, n_p, 9, 9), dtype=complex)
        iters[iface] = 0
        rel_res[iface] = np.zeros(n_freq, dtype=np.float64)

    # Build the (iface, freq_idx) job list.
    jobs: list[tuple[int, int, LayerSpec, list[MaterialContrast]]] = []
    for iface, layer, contrasts in pending_layers:
        for f_idx in range(n_freq):
            jobs.append((iface, f_idx, layer, contrasts))

    cache_dir_str = str(cache_dir_path) if cache_dir_path is not None else None
    use_parallel = n_workers is not None and n_workers >= 2 and len(jobs) >= 2

    if verbose:
        print(
            f"  τ-p mode: {len(pending_layers)} ifaces × {n_freq} freqs "
            f"= {len(jobs)} jobs (n_p={n_p} stacked as RHS columns)"
            + (f", n_workers={n_workers}" if use_parallel else " (sequential)")
        )

    if not use_parallel:
        for iface, f_idx, layer, contrasts in jobs:
            (
                _,
                _,
                _,
                T_block,
                its,
                rr,
                was_cached,
            ) = _solve_layer_freq_het_worker(
                iface=iface,
                freq_idx=f_idx,
                omega=complex(omegas[f_idx]),
                layer_name=layer.name,
                alpha=layer.alpha,
                beta=layer.beta,
                rho=layer.rho,
                M_vox=M_vox,
                N_z_vox=N_z_vox,
                a_km=a_km,
                slowness_in_2d=slowness_in_2d,
                slowness_out_2d=slowness_out_2d,
                contrasts=contrasts,
                rtol=config.block_gmres_rtol,
                max_iter=config.block_gmres_max_iter,
                cache_dir_str=cache_dir_str,
                force_recompute=force_recompute,
            )
            tmatrices[iface][f_idx] = T_block
            iters[iface] = max(iters[iface], int(its))
            rel_res[iface][f_idx] = rr
            if verbose:
                tag = "cached" if was_cached else "solved"
                print(
                    f"  iface {iface:2d} ({layer.name:>10s}) f[{f_idx:4d}]"
                    f"={float(omegas[f_idx].real):.3f}rad/s: "
                    f"iters={its:2d}, rel res={rr:.2e} [{tag}]"
                )
        return tmatrices, iters, rel_res

    n_effective = min(int(n_workers), len(jobs))  # type: ignore[arg-type]
    project_root = str(Path(__file__).resolve().parents[1])
    sibling_root = str(Path(project_root).parent / "MultipleScatteringCalculations")
    extra_paths = [project_root, sibling_root]
    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=n_effective,
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(extra_paths,),
    ) as executor:
        futures = {}
        for iface, f_idx, layer, contrasts in jobs:
            fut = executor.submit(
                _solve_layer_freq_het_worker,
                iface,
                f_idx,
                complex(omegas[f_idx]),
                layer.name,
                layer.alpha,
                layer.beta,
                layer.rho,
                M_vox,
                N_z_vox,
                a_km,
                slowness_in_2d,
                slowness_out_2d,
                contrasts,
                config.block_gmres_rtol,
                config.block_gmres_max_iter,
                cache_dir_str,
                force_recompute,
            )
            futures[fut] = (iface, f_idx, layer.name)

        completed = 0
        for fut in as_completed(futures):
            (
                iface,
                f_idx,
                _layer_name,
                T_block,
                its,
                rr,
                was_cached,
            ) = fut.result()
            tmatrices[iface][f_idx] = T_block
            iters[iface] = max(iters[iface], int(its))
            rel_res[iface][f_idx] = rr
            completed += 1
            if verbose:
                tag = "cached" if was_cached else "solved"
                print(
                    f"  [{completed:4d}/{len(jobs)}] iface {iface:2d} "
                    f"f[{f_idx:4d}]={float(omegas[f_idx].real):.3f}rad/s: "
                    f"iters={its:2d}, rel res={rr:.2e} [{tag}]"
                )

    tmatrices = {k: tmatrices[k] for k in sorted(tmatrices)}
    iters = {k: iters[k] for k in sorted(iters)}
    rel_res = {k: rel_res[k] for k in sorted(rel_res)}
    return tmatrices, iters, rel_res


# ---------------------------------------------------------------------------
# Outer Riccati sweep over ω × p
# ---------------------------------------------------------------------------


def compute_R_omega_p_stack(
    model: LayerModel,
    tmatrices_per_iface: dict[int, NDArray[np.complexfloating]],
    omegas: NDArray[np.complexfloating],
    slowness_2d: NDArray[np.floating],
    n_density: float,
) -> tuple[NDArray[np.complexfloating], NDArray[np.complexfloating]]:
    """Per-ω Riccati interlayer-MS sweep → ``R_total(ω, p)`` stack.

    The horizontal slowness grid is fully 2D: ``slowness_2d`` has shape
    ``(n_p, 2)`` and each row is an ``(p_x, p_y)`` pair in s/km.  At
    each ω the wavenumbers are
    ``kx = ω · p_x``, ``ky = ω · p_y``.  Single-azimuth τ-p along the
    x-axis is the special case ``slowness_2d[:, 1] = 0``.

    Three input shapes for ``tmatrices_per_iface[j]`` are accepted:

    * **3D ``(F, 9, 9)``** (k-space mode): one composite per ω, reused
      across all slownesses.  Physically inconsistent at oblique
      incidence but kept for the legacy normal-incidence path.
    * **4D ``(F, n_p, 9, 9)``** (τ-p mode, diagonal): one composite per
      ``(ω, p)``, no slowness coupling.  This is the contract of the
      previous τ-p path; new code should prefer the 5D shape below.
    * **5D ``(F, n_p_out, n_p_in, 9, 9)``** (τ-p mode, full coupling):
      the full slowness-coupling block produced by the slowness-batched
      inner solver.  Consumed by
      :func:`interlayer_ms_reflectivity_9x9_multi_p`, which solves the
      joint Foldy-Lax system coupling every slowness at every scatterer
      interface.  One multi-p solve per ω.

    Args:
        model: Background stratified elastic model.
        tmatrices_per_iface: Dict mapping iface → ``(F, 9, 9)``,
            ``(F, n_p, 9, 9)`` or ``(F, n_p_out, n_p_in, 9, 9)`` T-stack.
        omegas: Complex angular frequencies, shape ``(F,)``.
        slowness_2d: 2D horizontal slownesses in s/km, shape
            ``(n_p, 2)`` with columns ``(p_x, p_y)``.
        n_density: Areal number density (scatterers per unit area) —
            ``1 / (2 a)²`` for a space-filling lattice.

    Returns:
        Tuple ``(R_total, R_ref)`` each of shape ``(F, n_p)``.
        ``R_total`` includes the scatterers' response;
        ``R_ref`` is the background reflectivity of the layered model.
    """
    if not tmatrices_per_iface:
        msg = "tmatrices_per_iface is empty; cannot build ScattererSlab9x9"
        raise ValueError(msg)

    p_arr = np.asarray(slowness_2d, dtype=np.float64)
    if p_arr.ndim != 2 or p_arr.shape[1] != 2:
        msg = (
            f"slowness_2d has shape {p_arr.shape}, expected (n_p, 2) "
            "with columns (p_x, p_y)."
        )
        raise ValueError(msg)

    n_freq = omegas.shape[0]
    n_p = int(p_arr.shape[0])
    px = p_arr[:, 0]
    py = p_arr[:, 1]
    ifaces = sorted(tmatrices_per_iface.keys())

    # Detect 5D vs 4D vs 3D layout via the first iface; require all
    # ifaces to share the same layout.
    first_shape = tmatrices_per_iface[ifaces[0]].shape
    coupled_mode = False
    if len(first_shape) == 5:
        if first_shape != (n_freq, n_p, n_p, 9, 9):
            msg = (
                f"tmatrices_per_iface[{ifaces[0]}] has shape {first_shape}, "
                f"expected ({n_freq}, {n_p}, {n_p}, 9, 9) for slowness-"
                "coupled τ-p mode"
            )
            raise ValueError(msg)
        coupled_mode = True
        per_p_mode = True  # we still produce (F, n_p) output
    elif len(first_shape) == 4:
        per_p_mode = True
        if first_shape != (n_freq, n_p, 9, 9):
            msg = (
                f"tmatrices_per_iface[{ifaces[0]}] has shape {first_shape}, "
                f"expected ({n_freq}, {n_p}, 9, 9) for τ-p mode"
            )
            raise ValueError(msg)
    elif len(first_shape) == 3:
        per_p_mode = False
        if first_shape != (n_freq, 9, 9):
            msg = (
                f"tmatrices_per_iface[{ifaces[0]}] has shape {first_shape}, "
                f"expected ({n_freq}, 9, 9) for k-space mode"
            )
            raise ValueError(msg)
    else:
        msg = (
            f"tmatrices_per_iface[{ifaces[0]}] has ndim={len(first_shape)}; "
            "expected 3 (F, 9, 9), 4 (F, n_p, 9, 9), or "
            "5 (F, n_p_out, n_p_in, 9, 9)"
        )
        raise ValueError(msg)
    for j in ifaces[1:]:
        if tmatrices_per_iface[j].shape != first_shape:
            msg = (
                f"tmatrices_per_iface[{j}] shape {tmatrices_per_iface[j].shape} "
                f"differs from iface {ifaces[0]} shape {first_shape}"
            )
            raise ValueError(msg)

    R_total = np.zeros((n_freq, n_p), dtype=complex)
    R_ref = np.zeros((n_freq, n_p), dtype=complex)

    if not per_p_mode:
        # Legacy: one composite per ω, reused across all slownesses.
        for f_idx in range(n_freq):
            tmats_f = {j: tmatrices_per_iface[j][f_idx] for j in ifaces}
            slab = ScattererSlab9x9(
                model=model,
                scatterer_ifaces=ifaces,
                tmatrices=tmats_f,
                number_densities={j: n_density for j in ifaces},
            )
            om = complex(omegas[f_idx])
            kx = om.real * px
            ky = om.real * py
            res = interlayer_ms_reflectivity_9x9(slab, om, kx, ky)
            R_total[f_idx] = res.R_total
            R_ref[f_idx] = res.R_background
        return R_total, R_ref

    if coupled_mode:
        # 5D τ-p mode (full slowness coupling): one multi-p Riccati
        # solve per ω, consuming the entire (n_p, n_p, 9, 9) block.
        for f_idx in range(n_freq):
            om = complex(omegas[f_idx])
            tmats_f = {j: tmatrices_per_iface[j][f_idx] for j in ifaces}
            slab_mp = ScattererSlab9x9MultiP(
                model=model,
                scatterer_ifaces=ifaces,
                tmatrices=tmats_f,
                number_densities={j: n_density for j in ifaces},
            )
            kx = om.real * px
            ky = om.real * py
            res_mp = interlayer_ms_reflectivity_9x9_multi_p(slab_mp, om, kx, ky)
            R_total[f_idx] = res_mp.R_total
            R_ref[f_idx] = res_mp.R_background
        return R_total, R_ref

    # 4D τ-p mode (diagonal): build a fresh ScattererSlab9x9 per
    # (f_idx, p_idx) so the composite T at that (ω, p) is consumed by
    # interlayer_ms at that exact (ω, p).
    for f_idx in range(n_freq):
        om = complex(omegas[f_idx])
        for p_idx in range(n_p):
            tmats_fp = {j: tmatrices_per_iface[j][f_idx, p_idx] for j in ifaces}
            slab = ScattererSlab9x9(
                model=model,
                scatterer_ifaces=ifaces,
                tmatrices=tmats_fp,
                number_densities={j: n_density for j in ifaces},
            )
            kx = np.array([om.real * float(px[p_idx])])
            ky = np.array([om.real * float(py[p_idx])])
            res = interlayer_ms_reflectivity_9x9(slab, om, kx, ky)
            R_total[f_idx, p_idx] = res.R_total[0]
            R_ref[f_idx, p_idx] = res.R_background[0]

    return R_total, R_ref


# ---------------------------------------------------------------------------
# Ricker × IFFT → τ-p traces
# ---------------------------------------------------------------------------


def taup_traces_from_R_omega_p(
    R_omega_p: NDArray[np.complexfloating],
    omegas_real: NDArray[np.floating],
    T_record: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Synthesise τ-p traces from ``R(ω, p)`` via Ricker × Hermitian IFFT.

    Follows the exact convention of
    :func:`GlobalMatrix.taup_inversion.compute_taup_traces`: the
    ``(nw-1,)`` positive-frequency stack is zero-padded, mirrored with a
    complex conjugate, and transformed back to the time domain via
    ``np.fft.fft``.

    Args:
        R_omega_p: Positive-frequency reflectivity, shape
            ``(nw - 1, n_p)``.  Indices along ``axis=0`` map 1:1 to
            ``omegas_real`` (which excludes DC).
        omegas_real: Real angular frequencies, shape ``(nw - 1,)``,
            ``omegas_real[k] = (k + 1) dω`` with ``dω = 2π / T_record``.
        T_record: Record length in seconds.

    Returns:
        ``(time, traces)`` with ``time`` shape ``(nt,)`` where
        ``nt = 2 nw = 2 (len(omegas_real) + 1)``, and ``traces`` shape
        ``(nt, n_p)``, real-valued.
    """
    if R_omega_p.ndim != 2:
        msg = f"R_omega_p must be 2D (F, n_p); got ndim={R_omega_p.ndim}"
        raise ValueError(msg)
    nwm, n_p = R_omega_p.shape
    if omegas_real.shape != (nwm,):
        msg = (
            f"omegas_real shape {omegas_real.shape} inconsistent with "
            f"R_omega_p axis 0 = {nwm}"
        )
        raise ValueError(msg)
    nw = nwm + 1
    nt = 2 * nw
    dw = 2.0 * np.pi / T_record
    wmax = nw * dw

    S = ricker_spectrum(omegas_real, wmax)  # (nwm,)
    Y = R_omega_p * S[:, None]  # (nwm, n_p)

    U = np.zeros((nt, n_p), dtype=complex)
    U[1:nw, :] = Y
    U[nw + 1 :, :] = np.conj(Y[::-1, :])

    traces = np.real(np.fft.fft(U, axis=0))
    dt = T_record / float(nt)
    time = np.arange(nt, dtype=np.float64) * dt
    return time, traces


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_taup_wiggle_gather(
    time: NDArray[np.floating],
    traces: NDArray[np.floating],
    p_values: NDArray[np.floating],
    title: str,
    output: str | Path,
    *,
    t_max: float | None = None,
    clip: float = 0.8,
    normalize: str = "per_trace",
) -> None:
    """Save a τ-p variable-area wiggle gather to ``output``.

    Args:
        time: Time samples in seconds, shape ``(nt,)``.
        traces: Real-valued traces, shape ``(nt, n_p)``.
        p_values: Slowness values in s/km, shape ``(n_p,)``.
        title: Figure title.
        output: Output path (``.pdf`` recommended).
        t_max: Optional time-axis upper limit in seconds.
        clip: Variable-area clip fraction of the per-trace peak.
        normalize: ``"per_trace"`` (default) scales each trace by its own peak
            so every slowness is visible; ``"global"`` scales all traces by
            the same global peak to preserve relative amplitudes.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmask = time <= t_max if t_max is not None else np.ones_like(time, dtype=bool)
    t_plot = time[tmask]
    tr_plot = traces[tmask, :]

    if normalize not in {"per_trace", "global"}:
        raise ValueError(
            f"normalize must be 'per_trace' or 'global', got {normalize!r}"
        )

    if normalize == "global":
        global_peak = float(np.abs(tr_plot).max())
        if global_peak == 0.0:
            global_peak = 1.0
        peaks = np.full(tr_plot.shape[1], global_peak, dtype=np.float64)
    else:
        peaks = np.abs(tr_plot).max(axis=0).astype(np.float64)
        peaks[peaks == 0.0] = 1.0

    dp = float(p_values[1] - p_values[0]) if len(p_values) > 1 else 0.1

    fig, ax = plt.subplots(figsize=(10, 8))
    for j, p in enumerate(p_values):
        trace = tr_plot[:, j] / peaks[j]
        trace = np.clip(trace, -clip, clip)
        offset = float(p)
        ax.plot(offset + dp * trace, t_plot, "k-", linewidth=0.6)
        ax.fill_betweenx(
            t_plot,
            offset,
            offset + dp * trace,
            where=(trace > 0),
            color="k",
            linewidth=0.0,
        )

    ax.set_xlabel("Slowness p (s/km)")
    ax.set_ylabel("Time (s)")
    ax.set_title(title)
    ax.set_ylim(t_plot.max(), t_plot.min())  # time down
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_r_omega_p_heatmap(
    R_omega_p: NDArray[np.complexfloating],
    freqs_hz: NDArray[np.floating],
    p_values: NDArray[np.floating],
    title: str,
    output: str | Path,
    *,
    log_scale: bool = True,
    vmin_floor: float = 1.0e-3,
) -> None:
    """Save a ``|R(ω, p)|`` heatmap to ``output``.

    Args:
        R_omega_p: Reflectivity, shape ``(F, n_p)``.
        freqs_hz: Real frequencies in Hz, shape ``(F,)``.
        p_values: Slowness values in s/km, shape ``(n_p,)``.
        title: Figure title.
        output: Output path (``.pdf`` recommended).
        log_scale: If ``True``, render ``|R|`` on a log colour scale so
            scattering resonances and weak backgrounds are both visible.
        vmin_floor: Lower floor for the log colour scale (only used when
            ``log_scale`` is ``True``).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    amplitude = np.abs(R_omega_p)

    fig, ax = plt.subplots(figsize=(8, 6))
    extent = (
        float(p_values.min()),
        float(p_values.max()),
        float(freqs_hz.min()),
        float(freqs_hz.max()),
    )
    if log_scale:
        vmax = float(amplitude.max())
        vmin = max(float(vmin_floor), float(amplitude[amplitude > 0].min()))
        if vmax <= vmin:
            vmax = vmin * 10.0
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        im = ax.imshow(
            np.maximum(amplitude, vmin),
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap="viridis",
            norm=norm,
        )
    else:
        im = ax.imshow(
            amplitude,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap="viridis",
        )
    ax.set_xlabel("Slowness p (s/km)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=r"$|R(\omega, p)|$")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
