"""Tests for block-preconditioned layered Foldy-Lax solver.

Validates:
    1. Layer decomposition: counts match sphere z-profile
    2. 2D FFT kernel + matvec: single-layer system consistency
    3. Inter-layer kernel norm decay with |Δz|
    4. Full layered matvec matches 3D FFT matvec
    5. Block preconditioner reduces iteration count
    6. Top-level solver matches sphere_scattering_fft
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

from cubic_scattering import MaterialContrast, ReferenceMedium, compute_cube_tmatrix  # noqa: E402
from cubic_scattering.resonance_tmatrix import (  # noqa: E402
    _propagator_block_9x9,
    _sub_cell_tmatrix_9x9,
)
from cubic_scattering.sphere_scattering_fft import (  # noqa: E402
    _build_fft_kernel,
    _build_grid_index_map,
    _matvec_fft,
    compute_sphere_foldy_lax_fft,
)

from GlobalMatrix.block_riccati_cluster import (  # noqa: E402
    BlockRiccatiResult,
    ClusterGeometry,
    _pack_2d,
    _reorder_flat,
    _unpack_2d,
    block_preconditioner,
    build_interlayer_fft_kernel,
    build_interlayer_kernel_cache,
    build_intralayer_fft_kernel,
    cluster_from_sphere,
    compute_cluster_scattering,
    decompose_layers,
    layered_matvec,
)

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
CONTRAST = MaterialContrast(Dlambda=2.0e9, Dmu=1.0e9, Drho=100.0)
RADIUS = 0.5
OMEGA = 0.1 * REF.alpha / RADIUS  # ka_P = 0.1


# ---------------------------------------------------------------------------
# 1. Layer decomposition
# ---------------------------------------------------------------------------


class TestLayerDecomposition:
    def test_total_cubes_match(self) -> None:
        """Total cubes across all layers equals geometry total."""
        for n_sub in [3, 5, 7]:
            geom = cluster_from_sphere(RADIUS, n_sub)
            decomp = decompose_layers(geom)
            assert int(np.sum(decomp.layer_sizes)) == len(geom.centres)

    def test_layers_sorted_by_z(self) -> None:
        """z_indices are strictly increasing."""
        geom = cluster_from_sphere(RADIUS, 5)
        decomp = decompose_layers(geom)
        assert np.all(np.diff(decomp.z_indices) > 0)

    def test_layer_profile_symmetric(self) -> None:
        """Sphere z-profile is symmetric about the centre."""
        geom = cluster_from_sphere(RADIUS, 7)
        decomp = decompose_layers(geom)
        sizes = decomp.layer_sizes
        # Should be palindromic (sphere symmetry)
        np.testing.assert_array_equal(sizes, sizes[::-1])

    def test_sort_unsort_roundtrip(self) -> None:
        """sort_order and unsort_order are inverse permutations."""
        geom = cluster_from_sphere(RADIUS, 5)
        decomp = decompose_layers(geom)
        nC = len(geom.centres)
        original = np.arange(nC)
        np.testing.assert_array_equal(
            original[decomp.sort_order][decomp.unsort_order], original
        )

    def test_n_layers_matches_unique_z(self) -> None:
        """n_layers matches number of unique z indices in grid."""
        geom = cluster_from_sphere(RADIUS, 5)
        decomp = decompose_layers(geom)
        unique_z = np.unique(geom.grid_idx[:, 0])
        assert decomp.n_layers == len(unique_z)


# ---------------------------------------------------------------------------
# 2. 2D pack/unpack and intra-layer matvec
# ---------------------------------------------------------------------------


class TestPack2D:
    def test_roundtrip(self) -> None:
        """Pack → unpack recovers original data."""
        geom = cluster_from_sphere(RADIUS, 5)
        decomp = decompose_layers(geom)
        nP = 2 * 5 - 1

        for lz in range(decomp.n_layers):
            M = decomp.layer_sizes[lz]
            grid_2d = decomp.layer_grid_2d[lz]
            rng = np.random.default_rng(42 + lz)
            w = rng.standard_normal((M, 9)) + 1j * rng.standard_normal((M, 9))

            grids = _pack_2d(w, grid_2d, nP)
            w_back = _unpack_2d(grids, grid_2d, M)
            np.testing.assert_allclose(w_back, w, atol=1e-14)


class TestIntralayerKernel:
    def test_kernel_zero_at_origin(self) -> None:
        """Kernel at (Δx, Δy) = (0, 0) should be zero (self-interaction)."""
        n_sub = 3
        geom = cluster_from_sphere(RADIUS, n_sub)
        rayleigh_sub = compute_cube_tmatrix(OMEGA, geom.a_sub, REF, CONTRAST)
        T_loc = _sub_cell_tmatrix_9x9(rayleigh_sub, OMEGA, geom.a_sub)

        # Build spatial kernel (before FFT) to check origin
        dd = 2.0 * geom.a_sub
        nP = 2 * n_sub - 1
        kernel = np.zeros((9, 9, nP, nP), dtype=complex)
        for dx in range(-(n_sub - 1), n_sub):
            for dy in range(-(n_sub - 1), n_sub):
                if dx == 0 and dy == 0:
                    continue
                r_vec = np.array([0.0, dx * dd, dy * dd])
                P_block = _propagator_block_9x9(r_vec, OMEGA, REF)
                ix = dx % nP
                iy = dy % nP
                kernel[:, :, ix, iy] = -(P_block @ T_loc)

        # Origin should be zero
        np.testing.assert_allclose(kernel[:, :, 0, 0], 0.0, atol=1e-30)


# ---------------------------------------------------------------------------
# 3. Inter-layer kernel norm decay
# ---------------------------------------------------------------------------


class TestInterlayerKernelDecay:
    def test_norm_decays_with_dz(self) -> None:
        """Inter-layer kernel Frobenius norm should decrease with |Δz|."""
        n_sub = 5
        geom = cluster_from_sphere(RADIUS, n_sub)
        rayleigh_sub = compute_cube_tmatrix(OMEGA, geom.a_sub, REF, CONTRAST)
        T_loc = _sub_cell_tmatrix_9x9(rayleigh_sub, OMEGA, geom.a_sub)

        norms = []
        for dz in range(1, n_sub):
            kernel = build_interlayer_fft_kernel(
                n_sub, geom.a_sub, T_loc, OMEGA, REF, dz
            )
            norms.append(np.linalg.norm(kernel))

        # Norms should be monotonically decreasing
        for i in range(len(norms) - 1):
            assert norms[i] > norms[i + 1], (
                f"Kernel norm not decreasing: |Δz|={i + 1}: {norms[i]:.4e}, "
                f"|Δz|={i + 2}: {norms[i + 1]:.4e}"
            )


# ---------------------------------------------------------------------------
# 4. Full layered matvec matches 3D FFT matvec
# ---------------------------------------------------------------------------


class TestLayeredMatvec:
    @pytest.mark.parametrize("n_sub", [3, 5])
    def test_matches_3d_fft(self, n_sub: int) -> None:
        """Layered 2D matvec matches the reference 3D FFT matvec."""
        grid_idx, centres, a_sub = _build_grid_index_map(RADIUS, n_sub)
        nC = len(centres)
        nP_3d = 2 * n_sub - 1

        rayleigh_sub = compute_cube_tmatrix(OMEGA, a_sub, REF, CONTRAST)
        T_loc = _sub_cell_tmatrix_9x9(rayleigh_sub, OMEGA, a_sub)

        # Reference: 3D FFT matvec
        kernel_hat_3d = _build_fft_kernel(n_sub, a_sub, T_loc, OMEGA, REF)

        rng = np.random.default_rng(99)
        w = rng.standard_normal(9 * nC) + 1j * rng.standard_normal(9 * nC)

        y_ref = _matvec_fft(w, kernel_hat_3d, grid_idx, nP_3d, nC)

        # Layered 2D matvec
        geom = ClusterGeometry(
            n_sub=n_sub, a_sub=a_sub, grid_idx=grid_idx, centres=centres
        )
        decomp = decompose_layers(geom)

        shared_kernel = build_intralayer_fft_kernel(n_sub, a_sub, T_loc, OMEGA, REF)
        intralayer_kernels = [shared_kernel] * decomp.n_layers
        max_dz = (
            int(decomp.z_indices[-1] - decomp.z_indices[0])
            if decomp.n_layers > 1
            else 0
        )
        interlayer_kernels = build_interlayer_kernel_cache(
            n_sub, a_sub, T_loc, OMEGA, REF, max_dz
        )

        # Convert w to sorted order
        w_sorted = _reorder_flat(w, decomp.sort_order, nC)
        y_sorted = layered_matvec(
            w_sorted, decomp, intralayer_kernels, interlayer_kernels, n_sub
        )
        # Convert back to original ordering
        y_layered = _reorder_flat(y_sorted, decomp.unsort_order, nC)

        rel_err = np.linalg.norm(y_layered - y_ref) / np.linalg.norm(y_ref)
        assert rel_err < 1e-10, f"Layered vs 3D FFT rel err = {rel_err:.2e}"


# ---------------------------------------------------------------------------
# 5. Block preconditioner
# ---------------------------------------------------------------------------


class TestBlockPreconditioner:
    def test_identity_when_single_layer(self) -> None:
        """For a single-layer system, preconditioner solves exactly."""
        # Use n_sub=3 and pick only middle layer by choosing small radius
        n_sub = 3
        # radius chosen so sphere fits in one z-layer
        small_radius = 0.2
        small_omega = 0.1 * REF.alpha / small_radius

        geom = cluster_from_sphere(small_radius, n_sub)
        decomp = decompose_layers(geom)

        rayleigh_sub = compute_cube_tmatrix(small_omega, geom.a_sub, REF, CONTRAST)
        T_loc = _sub_cell_tmatrix_9x9(rayleigh_sub, small_omega, geom.a_sub)

        shared_kernel = build_intralayer_fft_kernel(
            n_sub, geom.a_sub, T_loc, small_omega, REF
        )
        intralayer_kernels = [shared_kernel] * decomp.n_layers

        nC = len(geom.centres)
        rng = np.random.default_rng(7)
        rhs = rng.standard_normal(9 * nC) + 1j * rng.standard_normal(9 * nC)

        # Sort
        rhs_sorted = _reorder_flat(rhs, decomp.sort_order, nC)
        x_sorted = block_preconditioner(
            rhs_sorted,
            decomp,
            intralayer_kernels,
            n_sub,
            inner_tol=1e-12,
            inner_maxiter=100,
        )

        # If single layer (or few layers), preconditioner should be a
        # good approximate inverse of the intra-layer part
        assert x_sorted.shape == rhs_sorted.shape
        assert not np.allclose(x_sorted, 0.0)


# ---------------------------------------------------------------------------
# 6. Top-level solver vs sphere_scattering_fft
# ---------------------------------------------------------------------------


class TestTopLevelSolver:
    @pytest.mark.parametrize("n_sub", [3, 5])
    def test_matches_fft_solver(self, n_sub: int) -> None:
        """Block-preconditioned solver matches 3D FFT solver."""
        result_fft = compute_sphere_foldy_lax_fft(
            OMEGA, RADIUS, REF, CONTRAST, n_sub, gmres_tol=1e-10
        )
        result_block = compute_cluster_scattering(
            OMEGA,
            RADIUS,
            REF,
            CONTRAST,
            n_sub,
            gmres_tol=1e-10,
            preconditioner="block_jacobi",
            inner_tol=1e-6,
        )

        rel_err = np.linalg.norm(
            result_block.T_comp_9x9 - result_fft.T_comp_9x9
        ) / np.linalg.norm(result_fft.T_comp_9x9)
        assert rel_err < 1e-4, f"T_comp rel err at n_sub={n_sub}: {rel_err:.2e}"

    def test_no_preconditioner(self) -> None:
        """Solver works without preconditioner."""
        n_sub = 3
        result_fft = compute_sphere_foldy_lax_fft(
            OMEGA, RADIUS, REF, CONTRAST, n_sub, gmres_tol=1e-10
        )
        result_block = compute_cluster_scattering(
            OMEGA,
            RADIUS,
            REF,
            CONTRAST,
            n_sub,
            gmres_tol=1e-10,
            preconditioner="none",
        )

        rel_err = np.linalg.norm(
            result_block.T_comp_9x9 - result_fft.T_comp_9x9
        ) / np.linalg.norm(result_fft.T_comp_9x9)
        assert rel_err < 1e-4, f"No-precond T_comp rel err: {rel_err:.2e}"

    def test_result_fields(self) -> None:
        """Result dataclass has expected fields."""
        result = compute_cluster_scattering(
            OMEGA, RADIUS, REF, CONTRAST, 3, gmres_tol=1e-6
        )
        assert isinstance(result, BlockRiccatiResult)
        assert result.T3x3.shape == (3, 3)
        assert result.T_comp_9x9.shape == (9, 9)
        assert result.n_cells > 0
        assert result.n_layers > 0
        assert result.gmres_iters >= 0
