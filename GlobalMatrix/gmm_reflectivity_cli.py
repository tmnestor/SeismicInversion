"""Compute GMM reflectivity and optionally compare with Kennett.

CLI entry point for the GlobalMatrix package. Computes plane-wave
reflectivity using the Global Matrix Method and (optionally) validates
against the Kennett recursive reflectivity.
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from Kennett_Reflectivity.kennett_reflectivity import kennett_reflectivity
from Kennett_Reflectivity.kennett_seismogram import default_ocean_crust_model

from .config import GMMConfig, load_config, save_config
from .global_matrix import gmm_reflectivity

__all__ = ["compute_and_compare", "main"]

logger = logging.getLogger(__name__)


def compute_and_compare(
    config: GMMConfig,
) -> dict[float, dict[str, np.ndarray]]:
    """Compute GMM reflectivity for all slownesses in the config.

    Args:
        config: GMM configuration.

    Returns:
        Dict mapping each slowness to a dict with keys:
        ``omega``, ``R_gmm``, and (if comparison enabled) ``R_kennett``.
    """
    T = 64.0
    nw = config.nfreq + 1
    dw = 2.0 * np.pi / T
    omega = np.arange(1, nw, dtype=np.float64) * dw

    do_comparison = "comparison" in config.output.formats
    results: dict[float, dict[str, np.ndarray]] = {}

    for p in config.p_values:
        logger.info(f"Computing GMM reflectivity for p={p:.4f}")
        R_gmm = gmm_reflectivity(
            config.model, p, omega, free_surface=config.free_surface
        )

        entry: dict[str, np.ndarray] = {"omega": omega, "R_gmm": R_gmm}

        if do_comparison:
            R_kennett = kennett_reflectivity(
                config.model, p, omega, free_surface=config.free_surface
            )
            entry["R_kennett"] = R_kennett
            diff = np.max(np.abs(R_gmm - R_kennett))
            logger.info(f"  p={p:.4f}: max|GMM - Kennett| = {diff:.2e}")

        results[p] = entry

    return results


def _plot_results(
    results: dict[float, dict[str, np.ndarray]],
    config: GMMConfig,
    output_path: Path,
) -> None:
    """Generate reflectivity and comparison plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    do_comparison = "comparison" in config.output.formats
    n_panels = len(config.p_values)
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3 * n_panels), squeeze=False)

    for i, p in enumerate(config.p_values):
        ax = axes[i, 0]
        entry = results[p]
        omega = entry["omega"]
        freq = omega / (2.0 * np.pi)

        ax.plot(freq, np.abs(entry["R_gmm"]), "b-", linewidth=0.8, label="GMM")
        if do_comparison and "R_kennett" in entry:
            ax.plot(
                freq,
                np.abs(entry["R_kennett"]),
                "r--",
                linewidth=0.8,
                label="Kennett",
            )
            ax.legend(fontsize=8)

        ax.set_ylabel("|R|")
        ax.set_title(f"p = {p:.2f} s/km")

    axes[-1, 0].set_xlabel("Frequency (Hz)")
    fig.suptitle(
        f"GMM Reflectivity — {config.model.n_layers}-layer model"
        f" ({'free surface' if config.free_surface else 'no free surface'})",
        fontsize=12,
    )
    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Plot saved to {output_path}")
    plt.close()


def _default_config() -> GMMConfig:
    """Build a default GMMConfig from the built-in 5-layer model."""
    model = default_ocean_crust_model()
    return GMMConfig(
        model=model,
        layer_names=["Ocean", "Sediment", "Crust", "Upper mantle", "Half-space"],
        fixed_layers=[0],
        p_values=[0.1, 0.2, 0.3, 0.4, 0.6],
    )


def main() -> None:
    """CLI entry point for GMM reflectivity computation."""
    parser = argparse.ArgumentParser(
        description="Compute GMM reflectivity and compare with Kennett.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Default model, compare with Kennett
  python -m GlobalMatrix.gmm_reflectivity_cli

  # From YAML config
  python -m GlobalMatrix.gmm_reflectivity_cli \\
      --config GlobalMatrix/configs/default_ocean_crust.yaml

  # Single slowness, no comparison
  python -m GlobalMatrix.gmm_reflectivity_cli -p 0.2 --no-compare
""",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config file (default: built-in model)",
    )
    parser.add_argument(
        "-p",
        "--slowness",
        type=float,
        nargs="+",
        default=None,
        help="Override slowness values (s/km)",
    )
    parser.add_argument(
        "-n",
        "--nfreq",
        type=int,
        default=None,
        help="Override number of frequencies",
    )
    parser.add_argument(
        "--free-surface",
        action="store_true",
        default=None,
        help="Include free surface reflections",
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Skip Kennett comparison",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="gmm_reflectivity.png",
        help="Output plot filename (default: gmm_reflectivity.png)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plot generation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load config
    if args.config is not None:
        config = load_config(args.config)
    else:
        config = _default_config()

    # CLI overrides
    if args.slowness is not None:
        config.p_values = args.slowness
    if args.nfreq is not None:
        config.nfreq = args.nfreq
    if args.free_surface is not None:
        config.free_surface = args.free_surface
    if args.no_compare:
        config.output.formats = ["reflectivity"]
    if args.output_dir is not None:
        config.output.directory = args.output_dir

    logger.info(
        f"Model: {config.model.n_layers} layers, "
        f"nfreq={config.nfreq}, "
        f"p_values={config.p_values}"
    )

    # Compute
    results = compute_and_compare(config)

    # Output
    out_dir = config.output.directory
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_plot:
        try:
            output_path = out_dir / args.output
            _plot_results(results, config, output_path)
        except ImportError:
            logger.warning("matplotlib not available, skipping plot")

    # Save data
    for p, entry in results.items():
        data_path = out_dir / f"gmm_reflectivity_p{p:.4f}.npz"
        np.savez(str(data_path), **entry)  # type: ignore[arg-type]
        logger.info(f"Data saved to {data_path}")

    # Save config for reproducibility
    config_path = out_dir / "gmm_config.yaml"
    save_config(config, config_path)
    logger.info(f"Config saved to {config_path}")


if __name__ == "__main__":
    main()
