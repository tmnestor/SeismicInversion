#!/usr/bin/env python3
"""
Example usage of Kennett_Reflectivity package.

Demonstrates:
1. Creating a custom layer model
2. Computing reflectivity at multiple slownesses
3. Generating synthetic seismograms
4. Analyzing scattering coefficients
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

# Load the package
spec = importlib.util.spec_from_file_location(
    "Kennett_Reflectivity",
    str(Path(__file__).parent / "__init__.py"),
)
if spec is None or spec.loader is None:
    msg = "Failed to locate Kennett_Reflectivity package"
    raise ImportError(msg)
pkg = importlib.util.module_from_spec(spec)
sys.modules["Kennett_Reflectivity"] = pkg
spec.loader.exec_module(pkg)

from Kennett_Reflectivity import (
    LayerModel,
    complex_slowness,
    compute_seismogram,
    default_ocean_crust_model,
    kennett_reflectivity,
    solid_solid_interface,
    vertical_slowness,
)


def example_1_default_model():
    """Example 1: Use default ocean-crust model."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Default Ocean-Crust Model")
    print("=" * 70)

    model = default_ocean_crust_model()

    print(f"\nModel has {model.n_layers} layers:")
    for i in range(model.n_layers):
        h = "∞" if np.isinf(model.thickness[i]) else f"{model.thickness[i]:.1f}"
        print(
            f"  Layer {i + 1}: "
            f"α={model.alpha[i]:.1f}, β={model.beta[i]:.1f}, "
            f"ρ={model.rho[i]:.1f}, h={h} m, "
            f"Q_α={model.Q_alpha[i]:.0f}, Q_β={model.Q_beta[i]:.0f}"
        )


def example_2_custom_model():
    """Example 2: Create a custom model."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Custom Model Creation")
    print("=" * 70)

    # Simple 3-layer model
    model = LayerModel.from_arrays(
        alpha=[2.0, 3.0, 4.0],
        beta=[1.0, 1.5, 2.0],
        rho=[2.5, 2.7, 2.9],
        thickness=[1.0, 2.0, np.inf],
        Q_alpha=[100, 100, 100],
        Q_beta=[50, 50, 50],
    )

    print(f"\nCreated custom model with {model.n_layers} layers")
    print(f"  Total depth: {np.sum(model.thickness[:-1]):.1f} m (excluding half-space)")


def example_3_reflectivity():
    """Example 3: Compute reflectivity at multiple slownesses."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Reflectivity at Multiple Slownesses")
    print("=" * 70)

    model = default_ocean_crust_model()
    omega = np.linspace(0.1, 10.0, 256)

    print(f"\nComputing reflectivity for {len(omega)} frequencies...")
    print(f"Frequency range: {omega[0]:.2f} to {omega[-1]:.2f} rad/s\n")

    slownesses = [0.1, 0.2, 0.3, 0.4]
    print(f"{'p (s/m)':>10} {'|R|_max':>12} {'|R|_min':>12} {'|R|_mean':>12}")
    print("-" * 50)

    for p in slownesses:
        R = kennett_reflectivity(model, p=p, omega=omega)
        abs_R = np.abs(R)
        print(
            f"{p:10.2f} {np.max(abs_R):12.6f} {np.min(abs_R):12.6f} "
            f"{np.mean(abs_R):12.6f}"
        )


def example_4_seismogram():
    """Example 4: Generate synthetic seismograms."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Seismogram Generation")
    print("=" * 70)

    model = default_ocean_crust_model()

    print("\nGenerating seismograms...")
    slownesses = [0.15, 0.25, 0.35]

    for p in slownesses:
        time, seismogram = compute_seismogram(
            model,
            p=p,
            T=32.0,
            nw=1024,
        )

        # Compute statistics
        rms = np.sqrt(np.mean(seismogram**2))
        peak = np.max(np.abs(seismogram))
        t_peak = time[np.argmax(np.abs(seismogram))]

        print(f"\n  p={p:.2f}:")
        print(f"    Duration: {time[-1]:.2f} s ({len(time)} samples)")
        print(f"    RMS amplitude: {rms:.6f}")
        print(f"    Peak amplitude: {peak:.6f} at t={t_peak:.2f} s")


def example_5_scattering():
    """Example 5: Analyze scattering coefficients."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Scattering Matrix Analysis")
    print("=" * 70)

    # Parameters for two interfaces
    p = 0.2

    print(f"\nHorizontal slowness: p = {p:.2f} s/m")
    print("\nComputing scattering coefficients for interfaces:")

    # Interface 1: Ocean-sediment (acoustic-elastic)
    print("\n1. Ocean-Sediment (Acoustic-Elastic):")
    eta1 = vertical_slowness(complex_slowness(1.5, 20000), p)
    eta2 = vertical_slowness(complex_slowness(1.6, 100), p)
    neta2 = vertical_slowness(complex_slowness(0.3, 100), p)

    from Kennett_Reflectivity import ocean_bottom_interface

    coeff_obs = ocean_bottom_interface(
        p=p,
        eta1=eta1,
        rho1=1.0,
        eta2=eta2,
        neta2=neta2,
        rho2=2.0,
        beta2=0.3,
    )

    print(f"  ||Rd|| = {np.linalg.norm(coeff_obs.Rd):.6f} (reflection)")
    print(f"  ||Ru|| = {np.linalg.norm(coeff_obs.Ru):.6f} (reflection)")
    print(f"  ||Tu|| = {np.linalg.norm(coeff_obs.Tu):.6f} (transmission)")
    print(f"  ||Td|| = {np.linalg.norm(coeff_obs.Td):.6f} (transmission)")

    # Interface 2: Sediment-Crust (elastic-elastic)
    print("\n2. Sediment-Crust (Elastic-Elastic):")
    eta3 = vertical_slowness(complex_slowness(3.0, 100), p)
    neta3 = vertical_slowness(complex_slowness(1.5, 100), p)

    coeff_ss = solid_solid_interface(
        p=p,
        eta1=eta2,
        neta1=neta2,
        rho1=2.0,
        beta1=0.3,
        eta2=eta3,
        neta2=neta3,
        rho2=3.0,
        beta2=1.5,
    )

    print(f"  ||Rd|| = {np.linalg.norm(coeff_ss.Rd):.6f} (reflection)")
    print(f"  ||Ru|| = {np.linalg.norm(coeff_ss.Ru):.6f} (reflection)")
    print(f"  ||Tu|| = {np.linalg.norm(coeff_ss.Tu):.6f} (transmission)")
    print(f"  ||Td|| = {np.linalg.norm(coeff_ss.Td):.6f} (transmission)")


def example_6_frequency_analysis():
    """Example 6: Analyze reflectivity vs frequency."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Frequency Response Analysis")
    print("=" * 70)

    model = default_ocean_crust_model()
    p = 0.25

    # Compute reflectivity at higher frequency resolution
    omega = np.logspace(-1, 1.5, 128)  # 0.1 to ~31 rad/s
    R = kennett_reflectivity(model, p=p, omega=omega)

    # Convert to Hz
    freq = omega / (2 * np.pi)

    print(f"\nReflectivity analysis for p={p:.2f}:")
    print(f"  Frequency range: {freq[0]:.3f} to {freq[-1]:.3f} Hz")

    # Find peaks
    abs_R = np.abs(R)
    peak_idx = np.argmax(abs_R)
    print(f"  Peak reflectivity: {abs_R[peak_idx]:.6f} at {freq[peak_idx]:.3f} Hz")

    # RMS by frequency band
    print("\n  RMS reflectivity by frequency band:")
    freq_bands = [(0.01, 0.1), (0.1, 1.0), (1.0, 10.0), (10.0, 100.0)]
    for f_min, f_max in freq_bands:
        mask = (freq >= f_min) & (freq <= f_max)
        if np.any(mask):
            rms = np.sqrt(np.mean(abs_R[mask] ** 2))
            print(f"    {f_min:.2f}–{f_max:.2f} Hz: {rms:.6f}")


def main():
    """Run all examples."""
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + " TMATRIX_DERIVATION USAGE EXAMPLES ".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    try:
        example_1_default_model()
        example_2_custom_model()
        example_3_reflectivity()
        example_4_seismogram()
        example_5_scattering()
        example_6_frequency_analysis()

        print("\n" + "█" * 70)
        print("█" + " " * 68 + "█")
        print("█" + " ✓ ALL EXAMPLES COMPLETED ".center(68) + "█")
        print("█" + " " * 68 + "█")
        print("█" * 70 + "\n")

        return 0

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
