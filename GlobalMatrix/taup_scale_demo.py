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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from cubic_scattering import MaterialContrast, ReferenceMedium
from Kennett_Reflectivity.layer_model import LayerModel
from Kennett_Reflectivity.source import ricker_spectrum

from .block_riccati_cluster import solve_slab_foldy_lax_freq_het
from .interlayer_ms import ScattererSlab9x9, interlayer_ms_reflectivity_9x9

__all__ = [
    "ScaleDemoConfig",
    "build_oceanic_crust_model",
    "compute_R_omega_p_stack",
    "compute_layer_composite_tmatrices_het",
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


def compute_layer_composite_tmatrices_het(
    config: ScaleDemoConfig,
    voxel_contrasts: dict[int, list[MaterialContrast]],
    omegas: NDArray[np.complexfloating],
    *,
    freq_batch_size: int | None = None,
    verbose: bool = False,
) -> tuple[
    dict[int, NDArray[np.complexfloating]],
    dict[int, int],
    dict[int, NDArray[np.floating]],
]:
    """Per-layer heterogeneous block Foldy-Lax → composite ``(F, 9, 9)``.

    For each elastic interface ``j``, constructs the layer's reference
    medium from :class:`LayerModel` properties and calls
    :func:`solve_slab_foldy_lax_freq_het` with the sampled per-cube
    contrasts.  Layers whose contrasts are all zero (e.g. from a user
    override forcing all Δλ=Δμ=Δρ=0) are skipped.

    When ``freq_batch_size`` is set, the frequency grid is split into
    chunks and the solver is called once per chunk, concatenating the
    per-chunk ``(F, 9, 9)`` outputs.  This keeps the block GMRES Krylov
    basis under the 4 GB per-call memory cap for large lattices (the
    cap scales as ``F · n_cubes · max_iter``).

    Args:
        config: Scale-demo configuration.
        voxel_contrasts: Dict mapping iface → per-cube contrasts.
        omegas: Complex angular frequencies, shape ``(F,)``.
        freq_batch_size: Optional maximum number of frequencies to
            process in one block GMRES call.  ``None`` means process
            all ``F`` frequencies in one go.
        verbose: If True, print per-layer iteration/residual summary.

    Returns:
        Tuple ``(tmatrices, iters, rel_res)`` where:

        * ``tmatrices[j]`` has shape ``(F, 9, 9)``.
        * ``iters[j]`` is the **maximum** number of block GMRES
          iterations across frequency batches.
        * ``rel_res[j]`` has shape ``(F,)`` giving per-ω relative
          residuals (concatenated over batches).
    """
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

    for iface, contrasts in voxel_contrasts.items():
        # Skip layers whose contrasts are all exactly zero — a way for
        # the user to force a particular layer out of the scattering
        # path without touching the solver.
        if all(c.Dlambda == 0.0 and c.Dmu == 0.0 and c.Drho == 0.0 for c in contrasts):
            if verbose:
                print(f"  iface {iface}: all-zero contrasts; skipping")
            continue

        layer = config.layers[iface]
        ref = ReferenceMedium(alpha=layer.alpha, beta=layer.beta, rho=layer.rho)

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
                rtol=config.block_gmres_rtol,
                max_iter=config.block_gmres_max_iter,
            )
            T_chunks.append(T_chunk)
            res_chunks.append(res_chunk)
            max_iters = max(max_iters, int(it_chunk))
            cursor += step

        T_freq = np.concatenate(T_chunks, axis=0)
        res = np.concatenate(res_chunks, axis=0)
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


# ---------------------------------------------------------------------------
# Outer Riccati sweep over ω × p
# ---------------------------------------------------------------------------


def compute_R_omega_p_stack(
    model: LayerModel,
    tmatrices_per_iface: dict[int, NDArray[np.complexfloating]],
    omegas: NDArray[np.complexfloating],
    p_values: NDArray[np.floating],
    n_density: float,
) -> tuple[NDArray[np.complexfloating], NDArray[np.complexfloating]]:
    """Per-ω Riccati interlayer-MS sweep → ``R_total(ω, p)`` stack.

    For each frequency, builds a :class:`ScattererSlab9x9` with the
    per-iface T-matrices at that ω and calls
    :func:`interlayer_ms_reflectivity_9x9` at ``(kx, ky) = (ω p, 0)``.

    Args:
        model: Background stratified elastic model.
        tmatrices_per_iface: Dict mapping iface → ``(F, 9, 9)`` T stack.
        omegas: Complex angular frequencies, shape ``(F,)``.
        p_values: Horizontal slownesses in s/km, shape ``(n_p,)``.
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

    n_freq = omegas.shape[0]
    n_p = p_values.shape[0]
    ifaces = sorted(tmatrices_per_iface.keys())

    R_total = np.zeros((n_freq, n_p), dtype=complex)
    R_ref = np.zeros((n_freq, n_p), dtype=complex)

    for f_idx in range(n_freq):
        tmats_f = {j: tmatrices_per_iface[j][f_idx] for j in ifaces}
        slab = ScattererSlab9x9(
            model=model,
            scatterer_ifaces=ifaces,
            tmatrices=tmats_f,
            number_densities={j: n_density for j in ifaces},
        )
        om = complex(omegas[f_idx])
        kx = om.real * p_values
        ky = np.zeros_like(kx)
        res = interlayer_ms_reflectivity_9x9(slab, om, kx, ky)
        R_total[f_idx] = res.R_total
        R_ref[f_idx] = res.R_background

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
