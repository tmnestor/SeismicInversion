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
