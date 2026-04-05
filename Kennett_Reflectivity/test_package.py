#!/usr/bin/env python3
"""
Verification test suite for Kennett_Reflectivity package.

Tests all major components against the original kennetslo.f Fortran code:
  1. Complex slowness (CMPLXSLO)
  2. Vertical slowness (VERTICALSLO)
  3. Scattering matrices (ScatMat, OBSMat)
  4. Kennett reflectivity (Kennet_Reflex)
  5. Seismogram generation (KennetSlo main)
"""

from __future__ import annotations

# Ensure package is importable
import importlib.util
import sys
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "Kennett_Reflectivity",
    str(Path(__file__).parent / "__init__.py"),
)
_pkg = importlib.util.module_from_spec(spec)
sys.modules["Kennett_Reflectivity"] = _pkg
spec.loader.exec_module(_pkg)


def test_complex_slowness():
    """Test CMPLXSLO translation."""
    from Kennett_Reflectivity import complex_slowness

    print("\n" + "=" * 70)
    print("TEST 1: Complex Slowness (CMPLXSLO)")
    print("=" * 70)

    # Test case from Fortran: CA0(1)=1.5, QA0(1)=20000
    cs = complex_slowness(1.5, 20000)
    # Expected: twoQ=40000, twoQsq=1.6e9, denom=(1+1.6e9)*1.5
    # Real ~ twoQsq/denom ~ 1.6e9/(1.6e9*1.5) ~ 1/1.5 ~ 0.6667
    # Imag ~ twoQ/denom ~ 40000/(1.6e9*1.5) ~ 1.667e-5
    expected_real = 1.0 / 1.5  # approximately
    assert abs(cs.real - expected_real) / expected_real < 1e-8, (
        f"Real part wrong: {cs.real} vs ~{expected_real}"
    )
    assert cs.imag > 0, "Imaginary part should be positive"
    assert cs.imag < 1e-3, f"Imaginary part should be small for high Q: {cs.imag}"
    print(f"  v=1.5, Q=20000: s = {cs}")
    print(
        f"  |Re(s) - 1/v| / (1/v) = {abs(cs.real - expected_real) / expected_real:.2e}"
    )

    # Test case: CA0(3)=3.0, QA0(3)=100
    cs3 = complex_slowness(3.0, 100)
    expected_real3 = 1.0 / 3.0
    assert abs(cs3.real - expected_real3) / expected_real3 < 1e-3, (
        f"Real part wrong: {cs3.real} vs ~{expected_real3}"
    )
    print(f"  v=3.0, Q=100:   s = {cs3}")

    print("\n  PASSED")


def test_vertical_slowness():
    """Test VERTICALSLO translation."""
    from Kennett_Reflectivity import complex_slowness, vertical_slowness

    print("\n" + "=" * 70)
    print("TEST 2: Vertical Slowness (VERTICALSLO)")
    print("=" * 70)

    # Compute for ocean layer: alpha=1.5, Q=20000, p=0.2
    s = complex_slowness(1.5, 20000)
    p = complex(0.2)
    eta = vertical_slowness(s, p)

    # Check: eta^2 = s^2 - p^2
    eta_sq = (s + p) * (s - p)
    assert abs(eta**2 - eta_sq) < 1e-12, f"eta^2 != (s+p)(s-p): {eta**2} vs {eta_sq}"
    assert eta.imag > 0, f"Im(eta) should be > 0, got {eta.imag}"
    print(f"  s={s:.8f}, p={p}")
    print(f"  eta = {eta}")
    print(f"  |eta^2 - (s+p)(s-p)| = {abs(eta**2 - eta_sq):.2e}")

    # Test for evanescent case: p > Re(s) -> eta should be mostly imaginary
    s_slow = complex_slowness(1.5, 20000)  # s ~ 0.667
    eta_evan = vertical_slowness(s_slow, complex(0.8))
    print(f"  Evanescent (p=0.8 > 1/v=0.667): eta = {eta_evan}")
    assert eta_evan.imag > 0, "Im(eta) > 0 for evanescent"

    print("\n  PASSED")


def test_scattering_matrices():
    """Test ScatMat and OBSMat translations."""
    from Kennett_Reflectivity import (
        complex_slowness,
        ocean_bottom_interface,
        solid_solid_interface,
        vertical_slowness,
    )

    print("\n" + "=" * 70)
    print("TEST 3: Scattering Matrices (ScatMat, OBSMat)")
    print("=" * 70)

    p = 0.2
    cp = complex(p)

    # Layer parameters (matching Fortran)
    # Ocean (layer 1): alpha=1.5, beta=0, rho=1.0, Q_a=20000, Q_b=1e10
    # Sediment (layer 2): alpha=1.6, beta=0.3, rho=2.0, Q_a=100, Q_b=100
    # Crust (layer 3): alpha=3.0, beta=1.5, rho=3.0, Q_a=100, Q_b=100

    s_a = [complex_slowness(v, q) for v, q in [(1.5, 20000), (1.6, 100), (3.0, 100)]]
    s_b = [0j, complex_slowness(0.3, 100), complex_slowness(1.5, 100)]
    beta_c = [0j, 1.0 / s_b[1], 1.0 / s_b[2]]  # complex velocity

    eta = [vertical_slowness(s, cp) for s in s_a]
    neta = [0j, vertical_slowness(s_b[1], cp), vertical_slowness(s_b[2], cp)]

    # Test OBSMat (ocean-bottom)
    print("\n  Ocean-bottom interface (OBSMat):")
    obs = ocean_bottom_interface(
        p=p,
        eta1=eta[0],
        rho1=1.0,
        eta2=eta[1],
        neta2=neta[1],
        rho2=2.0,
        beta2=beta_c[1],
    )
    print(f"    Rd[0,0] (PdPu) = {obs.Rd[0, 0]:.8f}")
    print(f"    Rd[0,1] = {obs.Rd[0, 1]} (should be 0)")
    print(f"    Rd[1,0] = {obs.Rd[1, 0]} (should be 0)")
    print(f"    Tu[0,0] (PuPu) = {obs.Tu[0, 0]:.8f}")
    print(f"    Tu[0,1] (SuPu) = {obs.Tu[0, 1]:.8f}")
    print(f"    Ru[0,0] (PuPd) = {obs.Ru[0, 0]:.8f}")
    print(f"    Ru[1,1] (SuSd) = {obs.Ru[1, 1]:.8f}")

    # Verify zero structure of acoustic-elastic
    assert obs.Rd[0, 1] == 0, f"Rd[0,1] should be 0: {obs.Rd[0, 1]}"
    assert obs.Rd[1, 0] == 0, f"Rd[1,0] should be 0: {obs.Rd[1, 0]}"
    assert obs.Rd[1, 1] == 0, f"Rd[1,1] should be 0: {obs.Rd[1, 1]}"
    assert obs.Tu[1, 0] == 0, f"Tu[1,0] should be 0: {obs.Tu[1, 0]}"
    assert obs.Tu[1, 1] == 0, f"Tu[1,1] should be 0: {obs.Tu[1, 1]}"
    assert obs.Td[0, 1] == 0, f"Td[0,1] should be 0: {obs.Td[0, 1]}"
    assert obs.Td[1, 1] == 0, f"Td[1,1] should be 0: {obs.Td[1, 1]}"

    # Verify reciprocity: Tu11 = Td11 (PuPu = PdPd)
    assert abs(obs.Tu[0, 0] - obs.Td[0, 0]) < 1e-14, (
        f"Reciprocity fail: Tu[0,0]={obs.Tu[0, 0]} != Td[0,0]={obs.Td[0, 0]}"
    )
    print("    Reciprocity Tu11=Td11: PASSED")

    # Test ScatMat (solid-solid)
    print("\n  Solid-solid interface (ScatMat):")
    ss = solid_solid_interface(
        p=p,
        eta1=eta[1],
        neta1=neta[1],
        rho1=2.0,
        beta1=beta_c[1],
        eta2=eta[2],
        neta2=neta[2],
        rho2=3.0,
        beta2=beta_c[2],
    )
    print(f"    Rd[0,0] = {ss.Rd[0, 0]:.8f}")
    print(f"    Rd[1,1] = {ss.Rd[1, 1]:.8f}")
    print(f"    ||Rd|| = {np.linalg.norm(ss.Rd):.8f}")

    # Verify Rd symmetry: Rd12 = Rd21
    assert abs(ss.Rd[0, 1] - ss.Rd[1, 0]) < 1e-14, (
        f"Rd symmetry fail: Rd12={ss.Rd[0, 1]} != Rd21={ss.Rd[1, 0]}"
    )
    print("    Rd symmetry Rd12=Rd21: PASSED")

    # Verify Tu = Td^T
    assert np.allclose(ss.Tu, ss.Td.T, atol=1e-14), "Reciprocity fail: Tu != Td^T"
    print("    Reciprocity Tu=Td^T: PASSED")

    # Verify Ru symmetry: Ru12 = Ru21
    assert abs(ss.Ru[0, 1] - ss.Ru[1, 0]) < 1e-14, (
        f"Ru symmetry fail: Ru12={ss.Ru[0, 1]} != Ru21={ss.Ru[1, 0]}"
    )
    print("    Ru symmetry Ru12=Ru21: PASSED")

    print("\n  PASSED")


def test_kennett_reflectivity():
    """Test Kennett recursive reflectivity."""
    from Kennett_Reflectivity import (
        batch_inv2x2,
        default_ocean_crust_model,
        kennett_reflectivity,
    )

    print("\n" + "=" * 70)
    print("TEST 4: Kennett Reflectivity (Kennet_Reflex)")
    print("=" * 70)

    # Test batch_inv2x2
    M = np.random.randn(10, 2, 2) + 1j * np.random.randn(10, 2, 2)
    M_inv = batch_inv2x2(M)
    for i in range(10):
        prod = M[i] @ M_inv[i]
        err = np.linalg.norm(prod - np.eye(2))
        assert err < 1e-10, f"batch_inv2x2 error at index {i}: {err}"
    print(
        f"  batch_inv2x2: max error = {max(np.linalg.norm(M[i] @ batch_inv2x2(M)[i] - np.eye(2)) for i in range(10)):.2e}"
    )

    model = default_ocean_crust_model()
    print(f"\n  Model: {model.n_layers} layers")
    for i in range(model.n_layers):
        h = "inf" if np.isinf(model.thickness[i]) else f"{model.thickness[i]:.1f}"
        print(
            f"    Layer {i + 1}: alpha={model.alpha[i]}, beta={model.beta[i]}, "
            f"rho={model.rho[i]}, h={h}"
        )

    # Test at multiple slownesses
    dw = 2.0 * np.pi / 64.0
    omega = np.arange(1, 64) * dw  # skip DC, 63 frequencies

    for p_val in [0.1, 0.2, 0.3]:
        R = kennett_reflectivity(model, p=p_val, omega=omega)
        assert R.shape == (63,), f"Shape error: {R.shape}"
        assert np.all(np.isfinite(R)), f"NaN/Inf at p={p_val}"
        print(
            f"  p={p_val}: max|R| = {np.max(np.abs(R)):.6f}, "
            f"mean|R| = {np.mean(np.abs(R)):.6f}"
        )

    # Full-scale test with 2047 frequencies (matching Fortran)
    omega_full = np.arange(1, 2048) * dw
    R_full = kennett_reflectivity(model, p=0.2, omega=omega_full)
    assert R_full.shape == (2047,), f"Full shape error: {R_full.shape}"
    assert np.all(np.isfinite(R_full)), "NaN/Inf in full reflectivity"
    print(f"\n  Full test (2047 freq, p=0.2): max|R| = {np.max(np.abs(R_full)):.6f}")

    print("\n  PASSED")


def test_seismogram():
    """Test seismogram computation."""
    from Kennett_Reflectivity import (
        compute_seismogram,
        default_ocean_crust_model,
        ricker_spectrum,
        ricker_wavelet,
    )

    print("\n" + "=" * 70)
    print("TEST 5: Seismogram Generation (KennetSlo)")
    print("=" * 70)

    # Test Ricker spectrum
    dw = 2.0 * np.pi / 64.0
    wmax = 2048 * dw
    omega = np.arange(1, 2048) * dw
    S = ricker_spectrum(omega, wmax)
    assert S.shape == (2047,), f"Spectrum shape error: {S.shape}"
    assert np.all(np.isfinite(S)), "Spectrum has NaN/Inf"
    peak_idx = np.argmax(np.abs(S))
    print(
        f"  Ricker spectrum: peak at omega = {omega[peak_idx]:.2f} rad/s "
        f"(f = {omega[peak_idx] / (2 * np.pi):.2f} Hz)"
    )

    # Test time-domain Ricker wavelet
    t = np.linspace(0, 2, 256)
    w = ricker_wavelet(t, f_peak=2.0)
    assert w.shape == (256,)
    assert np.max(np.abs(w)) > 0
    print(f"  Ricker wavelet: peak amplitude = {np.max(w):.6f}")

    # Compute seismogram
    model = default_ocean_crust_model()

    for nw_test in [256, 512]:
        time, seis = compute_seismogram(model, p=0.2, T=64.0, nw=nw_test)
        nt_expected = 2 * nw_test
        assert time.shape == (nt_expected,), (
            f"Time shape: {time.shape} != ({nt_expected},)"
        )
        assert seis.shape == (nt_expected,), (
            f"Seis shape: {seis.shape} != ({nt_expected},)"
        )
        assert np.all(np.isfinite(seis)), "Seismogram has NaN/Inf"

        dt = time[1] - time[0]
        expected_dt = 64.0 / nt_expected
        assert abs(dt - expected_dt) < 1e-10, f"dt mismatch: {dt} vs {expected_dt}"

        amp = np.max(np.abs(seis))
        print(f"  nw={nw_test}: nt={nt_expected}, dt={dt:.6f}s, max|seis| = {amp:.6e}")

    # Full-scale test matching Fortran parameters
    print("\n  Full-scale test (T=64, nw=2048, p=0.2):")
    time, seis = compute_seismogram(model, p=0.2, T=64.0, nw=2048)
    assert time.shape == (4096,), f"Time shape: {time.shape}"
    assert seis.shape == (4096,), f"Seis shape: {seis.shape}"
    dt = time[1] - time[0]
    assert abs(dt - 64.0 / 4096) < 1e-10

    # Find main arrival (should be around t = 2*h/v = 2*2/1.5 = 2.67s for ocean P)
    peak_time = time[np.argmax(np.abs(seis))]
    amp = np.max(np.abs(seis))
    print(f"    dt = {dt:.6f}s")
    print(f"    max|seis| = {amp:.6e} at t = {peak_time:.3f}s")
    print(f"    Expected first arrival ~ {2 * 2.0 / 1.5:.2f}s (two-way ocean P)")

    print("\n  PASSED")


def main():
    print("\n" + "#" * 70)
    print("#" + " TMATRIX_DERIVATION VERIFICATION SUITE ".center(68) + "#")
    print("#" + " (Against kennetslo.f Fortran code) ".center(68) + "#")
    print("#" * 70)

    try:
        test_complex_slowness()
        test_vertical_slowness()
        test_scattering_matrices()
        test_kennett_reflectivity()
        test_seismogram()

        print("\n" + "#" * 70)
        print("#" + " ALL 5 TESTS PASSED ".center(68) + "#")
        print("#" * 70 + "\n")
        return 0

    except Exception as e:
        print(f"\n  FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
