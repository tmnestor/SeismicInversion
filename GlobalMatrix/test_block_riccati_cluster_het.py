"""Tests for the heterogeneous-contrast block Foldy-Lax path.

Validates the ``*_het`` path added alongside the uniform-contrast
``solve_slab_foldy_lax_freq``:

    1. Canary: with a homogeneous list of per-cube contrasts, the
       heterogeneous driver reproduces the uniform driver elementwise
       to within ``rtol=1e-12, atol=1e-14``.  This is the primary
       correctness check: it proves that applying T in real space and
       then convolving by the propagator-only kernel is numerically
       lossless compared to folding T into the kernel.

    2. Heterogeneous soft+hard voxel mixture: the GMRES solve converges
       and returns a finite composite T.  Different seeds visibly
       change the composite T.

    3. Inner/outer composition: three heterogeneous inner composite
       T-matrices feed directly into
       ``interlayer_ms_reflectivity_9x9`` and give a finite
       ``R_total`` that differs from ``R_background``.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_MS_ROOT = Path("/Users/tod/Desktop/MultipleScatteringCalculations")
if str(_MS_ROOT) not in sys.path:
    sys.path.insert(0, str(_MS_ROOT))

from cubic_scattering import MaterialContrast, ReferenceMedium  # noqa: E402

from GlobalMatrix.block_riccati_cluster import (  # noqa: E402
    build_T_loc_per_cube_freq,
    cluster_from_slab,
    solve_slab_foldy_lax_freq,
    solve_slab_foldy_lax_freq_het,
    solve_slab_foldy_lax_single_freq_het_taup,
)
from GlobalMatrix.interlayer_ms import (  # noqa: E402
    ScattererSlab9x9,
    interlayer_ms_reflectivity_9x9,
)
from Kennett_Reflectivity.layer_model import LayerModel  # noqa: E402

# Units: km/s, g/cm³, km, GPa  (1 GPa ≡ (g/cm³)·(km/s)²)
REF = ReferenceMedium(alpha=3.0, beta=1.5, rho=2.5)
UNIFORM_CONTRAST = MaterialContrast(Dlambda=3.0, Dmu=1.5, Drho=0.1)

# Small shared frequency grid for the tests.
OMEGAS_TEST = 2 * np.pi * np.array([5.0, 10.0, 20.0]) + 0.1j


class TestHetCanary:
    """Canary: heterogeneous path with uniform contrasts == uniform path."""

    def test_het_matches_uniform_on_homogeneous_contrast(self) -> None:
        """``solve_slab_foldy_lax_freq_het`` reproduces
        ``solve_slab_foldy_lax_freq`` elementwise when every cube shares
        the same contrast."""
        M, N_z = 3, 2
        a = 10.0e-3  # km (= 10 m)
        nC = M * M * N_z

        T_uniform, iters_u, rel_u = solve_slab_foldy_lax_freq(
            M, N_z, a, OMEGAS_TEST, REF, UNIFORM_CONTRAST, rtol=1e-12, max_iter=60
        )

        contrasts = [UNIFORM_CONTRAST] * nC
        T_het, iters_h, rel_h = solve_slab_foldy_lax_freq_het(
            M, N_z, a, OMEGAS_TEST, REF, contrasts, rtol=1e-12, max_iter=60
        )

        assert T_het.shape == T_uniform.shape
        np.testing.assert_allclose(T_het, T_uniform, rtol=1e-12, atol=1e-14)
        # Both paths should converge in a comparable number of iterations.
        assert iters_h > 0
        assert iters_u > 0
        assert float(np.max(rel_h)) < 1e-6
        assert float(np.max(rel_u)) < 1e-6

    def test_build_T_loc_per_cube_uniform_shape_and_values(self) -> None:
        """Per-cube T builder with uniform contrast matches the scalar
        ``build_T_loc_freq`` along the cube axis."""
        from GlobalMatrix.block_riccati_cluster import build_T_loc_freq

        a = 10.0e-3
        nC = 5
        contrasts = [UNIFORM_CONTRAST] * nC
        T_per_cube = build_T_loc_per_cube_freq(a, REF, contrasts, OMEGAS_TEST)
        assert T_per_cube.shape == (len(OMEGAS_TEST), nC, 9, 9)

        T_scalar = build_T_loc_freq(a, REF, UNIFORM_CONTRAST, OMEGAS_TEST)
        for n in range(nC):
            np.testing.assert_allclose(
                T_per_cube[:, n], T_scalar, rtol=1e-14, atol=1e-16
            )


def _gmm_sample_contrast(
    rng: np.random.Generator,
    p_soft: float,
    hard_mean: tuple[float, float, float],
    hard_std: tuple[float, float, float],
    soft_mean: tuple[float, float, float],
    soft_std: tuple[float, float, float],
) -> MaterialContrast:
    """Draw a single 2-component Gaussian-mixture contrast."""
    if rng.uniform() < p_soft:
        mean, std = soft_mean, soft_std
    else:
        mean, std = hard_mean, hard_std
    return MaterialContrast(
        Dlambda=float(rng.normal(mean[0], std[0])),
        Dmu=float(rng.normal(mean[1], std[1])),
        Drho=float(rng.normal(mean[2], std[2])),
    )


class TestHetHeterogeneous:
    """End-to-end sanity on a genuinely heterogeneous slab."""

    def test_het_composite_T_finite_heterogeneous(self) -> None:
        """Random soft+hard mixture produces a finite, seed-sensitive T."""
        M, N_z = 4, 1
        a = 10.0e-3
        nC = M * M * N_z

        hard_mean = (2.0, 1.0, 0.1)
        hard_std = (0.3, 0.2, 0.02)
        soft_mean = (-1.5, -0.8, -0.15)
        soft_std = (0.2, 0.1, 0.02)
        p_soft = 0.25

        def _sample(seed: int) -> list[MaterialContrast]:
            rng = np.random.default_rng(seed)
            return [
                _gmm_sample_contrast(
                    rng, p_soft, hard_mean, hard_std, soft_mean, soft_std
                )
                for _ in range(nC)
            ]

        contrasts_a = _sample(20260406)
        T_a, iters_a, rel_a = solve_slab_foldy_lax_freq_het(
            M, N_z, a, OMEGAS_TEST, REF, contrasts_a, rtol=1e-8, max_iter=60
        )
        assert np.all(np.isfinite(T_a))
        assert iters_a > 0
        assert float(np.max(rel_a)) < 1e-6

        contrasts_b = _sample(99)
        T_b, _, rel_b = solve_slab_foldy_lax_freq_het(
            M, N_z, a, OMEGAS_TEST, REF, contrasts_b, rtol=1e-8, max_iter=60
        )
        assert np.all(np.isfinite(T_b))
        assert float(np.max(rel_b)) < 1e-6

        # Seed change must affect the composite T visibly.
        rel_diff = np.linalg.norm(T_a - T_b) / np.linalg.norm(T_a)
        assert rel_diff > 1e-3, (
            f"Composite T should depend on seed; got rel diff = {rel_diff:.2e}"
        )

    def test_het_cube_count_mismatch_raises(self) -> None:
        """Wrong-length contrasts list must fail fast."""
        M, N_z = 3, 2
        a = 10.0e-3
        with pytest.raises(ValueError, match="contrasts_per_cube has length"):
            solve_slab_foldy_lax_freq_het(
                M,
                N_z,
                a,
                OMEGAS_TEST,
                REF,
                [UNIFORM_CONTRAST] * 5,  # wrong length
                rtol=1e-8,
                max_iter=10,
            )

    def test_het_sort_order_matches_cluster_from_slab(self) -> None:
        """The contrasts list is consumed in ``cluster_from_slab`` order.

        Verifies that :func:`solve_slab_foldy_lax_freq_het` treats
        ``contrasts_per_cube[n]`` as the contrast for the ``n``-th cube
        in the canonical ``(iz outer, ix, iy inner)`` order.
        """
        M, N_z = 2, 2
        a = 10.0e-3
        nC = M * M * N_z
        geom = cluster_from_slab(M, N_z, a)
        assert geom.grid_idx.shape[0] == nC
        # Canonical order: iz outer, ix middle, iy inner.
        for n in range(nC):
            iz_expected = n // (M * M)
            ix_expected = (n // M) % M
            iy_expected = n % M
            iz, ix, iy = geom.grid_idx[n]
            assert (iz, ix, iy) == (iz_expected, ix_expected, iy_expected), (
                f"cluster_from_slab cube {n} index {(iz, ix, iy)} "
                f"does not match canonical {(iz_expected, ix_expected, iy_expected)}"
            )


class TestKParFreq:
    """The ``k_par_freq`` parameter dresses the incident basis at
    arbitrary horizontal slownesses (with complex ``kz`` for post-critical
    modes).  This is the path used by the per-slowness inner solver."""

    def test_k_par_freq_zero_p_matches_zhat_S(self) -> None:
        """``k_par_freq[f] = (0, 0, ω_f / β)`` reproduces the existing
        ``k_hat=z-hat, wave_type="S"`` baseline elementwise."""
        M, N_z = 3, 1
        a = 10.0e-3
        nC = M * M * N_z

        contrasts = [UNIFORM_CONTRAST] * nC

        # Baseline: existing k_hat=z, wave_type="S" path.
        T_base, _, _ = solve_slab_foldy_lax_freq_het(
            M, N_z, a, OMEGAS_TEST, REF, contrasts, rtol=1e-12, max_iter=60
        )

        # New path: explicit k_par_freq with kz = ω/β at every frequency.
        k_par_freq = np.zeros((len(OMEGAS_TEST), 3), dtype=complex)
        k_par_freq[:, 2] = OMEGAS_TEST / REF.beta
        T_kpar, _, _ = solve_slab_foldy_lax_freq_het(
            M,
            N_z,
            a,
            OMEGAS_TEST,
            REF,
            contrasts,
            k_par_freq=k_par_freq,
            rtol=1e-12,
            max_iter=60,
        )

        np.testing.assert_allclose(T_kpar, T_base, rtol=1e-12, atol=1e-14)

    def test_k_par_freq_oblique_changes_T(self) -> None:
        """Non-zero horizontal slowness must perturb the composite T."""
        M, N_z = 4, 1
        a = 10.0e-3
        nC = M * M * N_z
        rng = np.random.default_rng(7)
        contrasts = [
            _gmm_sample_contrast(
                rng,
                p_soft=0.2,
                hard_mean=(2.0, 1.0, 0.1),
                hard_std=(0.3, 0.2, 0.02),
                soft_mean=(-1.5, -0.8, -0.15),
                soft_std=(0.2, 0.1, 0.02),
            )
            for _ in range(nC)
        ]

        # Baseline at p=0 via k_par_freq.
        k_par_zero = np.zeros((len(OMEGAS_TEST), 3), dtype=complex)
        k_par_zero[:, 2] = OMEGAS_TEST / REF.beta
        T_zero, _, _ = solve_slab_foldy_lax_freq_het(
            M,
            N_z,
            a,
            OMEGAS_TEST,
            REF,
            contrasts,
            k_par_freq=k_par_zero,
            rtol=1e-10,
            max_iter=60,
        )

        # Oblique: p = 0.20 s/km along x, complex kz preserves k·k = (ω/β)².
        p_horiz = 0.20  # s/km
        k_par_obl = np.zeros((len(OMEGAS_TEST), 3), dtype=complex)
        for f_idx, om in enumerate(OMEGAS_TEST):
            kx = om * p_horiz
            kz_sq = (om / REF.beta) ** 2 - kx**2
            kz = np.sqrt(kz_sq)  # complex sqrt for post-critical
            k_par_obl[f_idx] = (kx, 0.0, kz)
        T_obl, _, _ = solve_slab_foldy_lax_freq_het(
            M,
            N_z,
            a,
            OMEGAS_TEST,
            REF,
            contrasts,
            k_par_freq=k_par_obl,
            rtol=1e-10,
            max_iter=60,
        )

        rel_diff = np.linalg.norm(T_obl - T_zero) / np.linalg.norm(T_zero)
        assert np.all(np.isfinite(T_obl))
        assert rel_diff > 1e-3, (
            f"Composite T should depend on horizontal slowness; "
            f"got rel diff = {rel_diff:.2e}"
        )

    def test_k_par_freq_shape_validation(self) -> None:
        """Wrong-shape ``k_par_freq`` must fail fast."""
        M, N_z = 2, 1
        a = 10.0e-3
        nC = M * M * N_z
        contrasts = [UNIFORM_CONTRAST] * nC
        bad = np.zeros((len(OMEGAS_TEST), 2), dtype=complex)  # should be 3
        with pytest.raises(ValueError, match="k_par_freq has shape"):
            solve_slab_foldy_lax_freq_het(
                M,
                N_z,
                a,
                OMEGAS_TEST,
                REF,
                contrasts,
                k_par_freq=bad,
                rtol=1e-8,
                max_iter=10,
            )


class TestKParOutFreqProjection:
    """The ``k_par_out_freq`` parameter projects per-cube scattered
    moments onto outgoing plane waves and produces an
    ``(F, n_p_out, 9, 9)`` slowness-coupling row instead of the legacy
    single ``(F, 9, 9)`` composite."""

    def test_projection_zero_kpar_out_sums_moments(self) -> None:
        """At ``k_par_out_freq = 0`` the projected output reduces to the
        legacy unprojected sum (``exp(-i 0 · x_n) = 1`` for every cube)."""
        M, N_z = 3, 1
        a = 10.0e-3
        nC = M * M * N_z
        contrasts = [UNIFORM_CONTRAST] * nC

        T_legacy, _, _ = solve_slab_foldy_lax_freq_het(
            M, N_z, a, OMEGAS_TEST, REF, contrasts, rtol=1e-12, max_iter=60
        )  # (F, 9, 9)

        # Single output slowness with zero wave-vector at every freq.
        k_par_out = np.zeros((len(OMEGAS_TEST), 1, 3), dtype=complex)
        T_proj, _, _ = solve_slab_foldy_lax_freq_het(
            M,
            N_z,
            a,
            OMEGAS_TEST,
            REF,
            contrasts,
            k_par_out_freq=k_par_out,
            rtol=1e-12,
            max_iter=60,
        )  # (F, 1, 9, 9)

        assert T_proj.shape == (len(OMEGAS_TEST), 1, 9, 9)
        np.testing.assert_allclose(T_proj[:, 0], T_legacy, rtol=1e-12, atol=1e-14)

    def test_projection_multiple_outgoing_slownesses_finite(self) -> None:
        """A 4-slowness output stack returns a finite ``(F, 4, 9, 9)``
        coupling row, and the rows for distinct slownesses differ."""
        M, N_z = 4, 1
        a = 10.0e-3
        nC = M * M * N_z
        rng = np.random.default_rng(31)
        contrasts = [
            _gmm_sample_contrast(
                rng,
                p_soft=0.2,
                hard_mean=(2.0, 1.0, 0.1),
                hard_std=(0.3, 0.2, 0.02),
                soft_mean=(-1.5, -0.8, -0.15),
                soft_std=(0.2, 0.1, 0.02),
            )
            for _ in range(nC)
        ]

        # Incident slowness p_in = 0.10 s/km along x.
        p_in = 0.10
        k_par_in = np.zeros((len(OMEGAS_TEST), 3), dtype=complex)
        for f_idx, om in enumerate(OMEGAS_TEST):
            kx = om * p_in
            kz = np.sqrt((om / REF.beta) ** 2 - kx**2)
            k_par_in[f_idx] = (kx, 0.0, kz)

        # Four outgoing slownesses, well-separated so the projection
        # phase varies measurably across the small test lattice
        # (M·2a = 80 m at the test scale).
        p_out_values = np.array([0.02, 0.20, 0.40, 0.55])
        n_p_out = len(p_out_values)
        k_par_out = np.zeros((len(OMEGAS_TEST), n_p_out, 3), dtype=complex)
        for f_idx, om in enumerate(OMEGAS_TEST):
            for p_idx, p in enumerate(p_out_values):
                kx = om * p
                kz = np.sqrt((om / REF.beta) ** 2 - kx**2)
                k_par_out[f_idx, p_idx] = (kx, 0.0, kz)

        T_row, _, rel = solve_slab_foldy_lax_freq_het(
            M,
            N_z,
            a,
            OMEGAS_TEST,
            REF,
            contrasts,
            k_par_freq=k_par_in,
            k_par_out_freq=k_par_out,
            rtol=1e-10,
            max_iter=60,
        )

        assert T_row.shape == (len(OMEGAS_TEST), n_p_out, 9, 9)
        assert np.all(np.isfinite(T_row))
        assert float(np.max(rel)) < 1e-6

        # Different outgoing slownesses must give different projected
        # composites (the cluster has no symmetry that would make any
        # two outgoing slownesses identical).
        for i in range(n_p_out):
            for j in range(i + 1, n_p_out):
                rel_diff = np.linalg.norm(T_row[:, i] - T_row[:, j]) / np.linalg.norm(
                    T_row[:, i]
                )
                assert rel_diff > 1e-3, (
                    f"Outgoing slownesses {p_out_values[i]:.3f} and "
                    f"{p_out_values[j]:.3f} produced identical projections "
                    f"(rel diff = {rel_diff:.2e})"
                )

    def test_projection_shape_validation(self) -> None:
        """Wrong-shape ``k_par_out_freq`` must fail fast."""
        M, N_z = 2, 1
        a = 10.0e-3
        nC = M * M * N_z
        contrasts = [UNIFORM_CONTRAST] * nC

        # Wrong frequency axis.
        bad = np.zeros((len(OMEGAS_TEST) + 1, 2, 3), dtype=complex)
        with pytest.raises(ValueError, match="k_par_out_freq has shape"):
            solve_slab_foldy_lax_freq_het(
                M,
                N_z,
                a,
                OMEGAS_TEST,
                REF,
                contrasts,
                k_par_out_freq=bad,
                rtol=1e-8,
                max_iter=10,
            )

        # Wrong wave-vector dimensionality.
        bad2 = np.zeros((len(OMEGAS_TEST), 2, 2), dtype=complex)
        with pytest.raises(ValueError, match="third axis must be size 3"):
            solve_slab_foldy_lax_freq_het(
                M,
                N_z,
                a,
                OMEGAS_TEST,
                REF,
                contrasts,
                k_par_out_freq=bad2,
                rtol=1e-8,
                max_iter=10,
            )


class TestSlownessBatchedTaup:
    """``solve_slab_foldy_lax_single_freq_het_taup``: at fixed ω all
    incident slownesses become extra RHS columns of one block-GMRES,
    and the per-cube moments are projected onto every outgoing
    slowness to produce the full ``(n_p_out, n_p_in, 9, 9)`` coupling
    matrix in a single solve."""

    def test_single_pin_pout_zero_matches_legacy(self) -> None:
        """One incident slowness ``(0, 0, ω/β)`` and one ``k_par_out = 0``
        must reproduce the legacy het ``(F, 9, 9)`` composite at the
        same single ω, bit-identically."""
        M, N_z = 3, 1
        a = 10.0e-3
        nC = M * M * N_z
        rng = np.random.default_rng(7)
        contrasts = [
            _gmm_sample_contrast(
                rng,
                p_soft=0.2,
                hard_mean=(2.0, 1.0, 0.1),
                hard_std=(0.3, 0.2, 0.02),
                soft_mean=(-1.5, -0.8, -0.15),
                soft_std=(0.2, 0.1, 0.02),
            )
            for _ in range(nC)
        ]
        omega_c = OMEGAS_TEST[1]  # 2π·10 Hz + 0.1j

        T_legacy, _, _ = solve_slab_foldy_lax_freq_het(
            M,
            N_z,
            a,
            np.array([omega_c]),
            REF,
            contrasts,
            k_hat=np.array([0.0, 0.0, 1.0]),
            wave_type="S",
            rtol=1e-12,
            max_iter=60,
        )  # (1, 9, 9)

        k_par_in = np.array([[0.0, 0.0, omega_c / REF.beta]], dtype=complex)
        k_par_out = np.zeros((1, 3), dtype=complex)
        T_new, _, rel = solve_slab_foldy_lax_single_freq_het_taup(
            M=M,
            N_z=N_z,
            a=a,
            omega=complex(omega_c),
            ref=REF,
            contrasts_per_cube=contrasts,
            k_par_in=k_par_in,
            k_par_out=k_par_out,
            rtol=1e-12,
            max_iter=60,
        )

        assert T_new.shape == (1, 1, 9, 9)
        assert rel < 1e-6
        np.testing.assert_allclose(T_new[0, 0], T_legacy[0], rtol=1e-12, atol=1e-14)

    def test_multi_pin_matches_per_pin_freq_solver(self) -> None:
        """Stacking ``n_p_in = 3`` slownesses as RHS columns of one
        slowness-batched solve gives the same row of composites as
        calling the freq-batched solver three times, once per slowness,
        with matching ``k_par_freq`` and ``k_par_out_freq`` rows."""
        M, N_z = 3, 1
        a = 10.0e-3
        nC = M * M * N_z
        rng = np.random.default_rng(11)
        contrasts = [
            _gmm_sample_contrast(
                rng,
                p_soft=0.2,
                hard_mean=(1.5, 0.8, 0.08),
                hard_std=(0.25, 0.15, 0.02),
                soft_mean=(-1.2, -0.6, -0.12),
                soft_std=(0.2, 0.1, 0.02),
            )
            for _ in range(nC)
        ]
        omega_c = OMEGAS_TEST[1]
        omegas_single = np.array([omega_c], dtype=complex)

        # Three incident slownesses, well-separated.
        p_in_values = np.array([0.05, 0.20, 0.45])
        k_par_in = np.zeros((len(p_in_values), 3), dtype=complex)
        for k, p in enumerate(p_in_values):
            kx = omega_c * p
            kz = np.sqrt((omega_c / REF.beta) ** 2 - kx**2)
            k_par_in[k] = (kx, 0.0, kz)

        # Two outgoing slownesses to project onto.
        p_out_values = np.array([0.02, 0.35])
        k_par_out = np.zeros((len(p_out_values), 3), dtype=complex)
        for k, p in enumerate(p_out_values):
            kx = omega_c * p
            kz = np.sqrt((omega_c / REF.beta) ** 2 - kx**2)
            k_par_out[k] = (kx, 0.0, kz)

        # Slowness-batched single-frequency solve: one block GMRES.
        T_batched, iters, rel = solve_slab_foldy_lax_single_freq_het_taup(
            M=M,
            N_z=N_z,
            a=a,
            omega=complex(omega_c),
            ref=REF,
            contrasts_per_cube=contrasts,
            k_par_in=k_par_in,
            k_par_out=k_par_out,
            rtol=1e-12,
            max_iter=80,
        )
        assert T_batched.shape == (len(p_out_values), len(p_in_values), 9, 9)
        assert rel < 1e-6

        # Reference: per-slowness call to the freq-batched solver, with
        # the same k_par_in[k] dressed onto the incident field and the
        # full k_par_out stack as the projection target.
        k_par_out_freq_full = np.broadcast_to(
            k_par_out[None, :, :], (1, len(p_out_values), 3)
        ).copy()
        for k_idx in range(len(p_in_values)):
            k_par_freq_one = k_par_in[k_idx][None, :]  # (1, 3)
            T_ref_row, _, rel_ref = solve_slab_foldy_lax_freq_het(
                M,
                N_z,
                a,
                omegas_single,
                REF,
                contrasts,
                k_par_freq=k_par_freq_one,
                k_par_out_freq=k_par_out_freq_full,
                rtol=1e-12,
                max_iter=80,
            )
            assert float(np.max(rel_ref)) < 1e-6
            # T_ref_row has shape (1, n_p_out, 9, 9); compare against
            # the corresponding p_in column of the batched output.
            np.testing.assert_allclose(
                T_batched[:, k_idx],
                T_ref_row[0],
                rtol=1e-10,
                atol=1e-12,
                err_msg=(
                    f"slowness-batched column {k_idx} does not match "
                    "per-slowness reference"
                ),
            )

    def test_distinct_slownesses_distinct_couplings(self) -> None:
        """A 3x3 coupling block has distinct rows and distinct columns
        for distinct slownesses."""
        M, N_z = 4, 1
        a = 10.0e-3
        nC = M * M * N_z
        rng = np.random.default_rng(19)
        contrasts = [
            _gmm_sample_contrast(
                rng,
                p_soft=0.15,
                hard_mean=(2.0, 1.0, 0.1),
                hard_std=(0.3, 0.2, 0.02),
                soft_mean=(-1.5, -0.8, -0.15),
                soft_std=(0.2, 0.1, 0.02),
            )
            for _ in range(nC)
        ]
        omega_c = complex(OMEGAS_TEST[2])

        slownesses = np.array([0.02, 0.20, 0.45])
        k_par = np.zeros((len(slownesses), 3), dtype=complex)
        for k, p in enumerate(slownesses):
            kx = omega_c * p
            kz = np.sqrt((omega_c / REF.beta) ** 2 - kx**2)
            k_par[k] = (kx, 0.0, kz)

        T_block, _, rel = solve_slab_foldy_lax_single_freq_het_taup(
            M=M,
            N_z=N_z,
            a=a,
            omega=omega_c,
            ref=REF,
            contrasts_per_cube=contrasts,
            k_par_in=k_par,
            k_par_out=k_par,
            rtol=1e-10,
            max_iter=80,
        )
        assert T_block.shape == (3, 3, 9, 9)
        assert np.all(np.isfinite(T_block))
        assert rel < 1e-6

        # Distinct rows (different p_out, fixed p_in).
        for i in range(3):
            for j in range(i + 1, 3):
                rd = np.linalg.norm(T_block[i, 0] - T_block[j, 0]) / np.linalg.norm(
                    T_block[i, 0]
                )
                assert rd > 1e-3, f"p_out rows {i},{j} identical (rd={rd:.2e})"
        # Distinct columns (different p_in, fixed p_out).
        for i in range(3):
            for j in range(i + 1, 3):
                rd = np.linalg.norm(T_block[0, i] - T_block[0, j]) / np.linalg.norm(
                    T_block[0, i]
                )
                assert rd > 1e-3, f"p_in cols {i},{j} identical (rd={rd:.2e})"

    def test_shape_validation(self) -> None:
        """Wrong shapes for ``k_par_in`` / ``k_par_out`` must fail fast."""
        M, N_z = 2, 1
        a = 10.0e-3
        nC = M * M * N_z
        contrasts = [UNIFORM_CONTRAST] * nC
        omega_c = complex(OMEGAS_TEST[0])

        bad_in = np.zeros((2, 2), dtype=complex)
        with pytest.raises(ValueError, match="k_par_in has shape"):
            solve_slab_foldy_lax_single_freq_het_taup(
                M=M,
                N_z=N_z,
                a=a,
                omega=omega_c,
                ref=REF,
                contrasts_per_cube=contrasts,
                k_par_in=bad_in,
                k_par_out=np.zeros((1, 3), dtype=complex),
                rtol=1e-8,
                max_iter=10,
            )

        bad_out = np.zeros((1, 4), dtype=complex)
        with pytest.raises(ValueError, match="k_par_out has shape"):
            solve_slab_foldy_lax_single_freq_het_taup(
                M=M,
                N_z=N_z,
                a=a,
                omega=omega_c,
                ref=REF,
                contrasts_per_cube=contrasts,
                k_par_in=np.zeros((1, 3), dtype=complex),
                k_par_out=bad_out,
                rtol=1e-8,
                max_iter=10,
            )


class TestHetInnerOuterCompose:
    """The heterogeneous inner solver composes with the outer Riccati path."""

    def test_het_inner_outer_composes(self) -> None:
        """Het composite T feeds cleanly into ``interlayer_ms_reflectivity_9x9``."""
        # Simple 4-layer marine model: water + two elastic layers + mantle.
        model = LayerModel(
            alpha=np.array([1.5, 2.0, 3.5, 6.5]),
            beta=np.array([0.0, 1.0, 1.9, 3.7]),
            rho=np.array([1.0, 2.0, 2.5, 3.0]),
            thickness=np.array([2.0, 0.3, 0.4, np.inf]),
            Q_alpha=np.array([10000.0, 200.0, 300.0, 500.0]),
            Q_beta=np.array([10000.0, 100.0, 150.0, 250.0]),
        )

        # Tiny slab parameters so the test stays fast.
        M, N_z = 3, 1
        a = 5.0e-3  # 5 m cubes
        nC = M * M * N_z
        omega_single = np.array([2 * np.pi * 8.0 + 0.1j], dtype=complex)

        rng = np.random.default_rng(42)
        iface_refs = {
            1: ReferenceMedium(alpha=2.0, beta=1.0, rho=2.0),
            2: ReferenceMedium(alpha=3.5, beta=1.9, rho=2.5),
        }

        tmatrices_by_iface: dict[int, np.ndarray] = {}
        for iface, ref in iface_refs.items():
            contrasts = [
                _gmm_sample_contrast(
                    rng,
                    p_soft=0.2,
                    hard_mean=(1.5, 0.8, 0.08),
                    hard_std=(0.2, 0.15, 0.02),
                    soft_mean=(-1.0, -0.5, -0.1),
                    soft_std=(0.15, 0.1, 0.02),
                )
                for _ in range(nC)
            ]
            T_freq, _, rel = solve_slab_foldy_lax_freq_het(
                M,
                N_z,
                a,
                omega_single,
                ref,
                contrasts,
                rtol=1e-8,
                max_iter=60,
            )
            assert float(np.max(rel)) < 1e-6
            assert np.all(np.isfinite(T_freq))
            tmatrices_by_iface[iface] = T_freq[0]  # (9, 9) for the single ω

        slab = ScattererSlab9x9(
            model=model,
            scatterer_ifaces=list(tmatrices_by_iface.keys()),
            tmatrices=tmatrices_by_iface,
            number_densities={j: 1.0 / (2.0 * a) ** 2 for j in tmatrices_by_iface},
        )

        p_values = np.array([0.05, 0.15, 0.25])  # s/km
        om = complex(omega_single[0])
        kx = om.real * p_values
        ky = np.zeros_like(kx)
        res = interlayer_ms_reflectivity_9x9(slab, om, kx, ky)

        assert res.R_total.shape == p_values.shape
        assert res.R_background.shape == p_values.shape
        assert np.all(np.isfinite(res.R_total))
        assert np.all(np.isfinite(res.R_background))
        # The scatterers must perturb the background.
        rel_pert = np.linalg.norm(res.R_total - res.R_background) / np.linalg.norm(
            res.R_background
        )
        assert rel_pert > 1e-4, (
            f"Het scatterers did not perturb R_background; rel pert = {rel_pert:.2e}"
        )
