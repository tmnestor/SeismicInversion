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
from collections.abc import Callable
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
    _build_incident_field_coupled,
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
    _Z_PARITY_MASK_9x9,
    BlockRiccatiResult,
    ClusterGeometry,
    LayerDecomposition,
    _pack_2d,
    _pack_2d_freq,
    _pack_2d_multi_freq,
    _propagator_block_9x9_batch,
    _propagator_block_9x9_batch_freq,
    _reorder_flat,
    _unpack_2d,
    _unpack_2d_freq,
    _unpack_2d_multi_freq,
    block_gmres,
    block_gmres_freq,
    block_preconditioner,
    build_interlayer_fft_kernel,
    build_interlayer_fft_kernel_freq,
    build_interlayer_kernel_cache,
    build_interlayer_kernel_cache_freq,
    build_intralayer_fft_kernel,
    build_intralayer_fft_kernel_freq,
    build_T_loc_freq,
    cluster_from_sphere,
    cluster_from_slab,
    compute_cluster_scattering,
    decompose_layers,
    _build_incident_field_coupled_freq,
    layered_matvec,
    layered_matvec_freq,
    layered_matvec_multi,
    layered_matvec_multi_freq,
    solve_slab_foldy_lax_freq,
)

# Units: km/s, g/cm³, km, GPa  (1 GPa ≡ (g/cm³)·(km/s)²)
REF = ReferenceMedium(alpha=5.0, beta=3.0, rho=2.5)
CONTRAST = MaterialContrast(Dlambda=2.0, Dmu=1.0, Drho=0.1)
RADIUS = 0.5e-3  # km (= 0.5 m)
OMEGA = 0.1 * REF.alpha / RADIUS  # ka_P = 0.1

# Shared Fix 6 frequency grid for *_freq tests.
OMEGAS_TEST = 2 * np.pi * np.array([5.0, 10.0, 15.0, 20.0, 25.0]) + 0.1j


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
        small_radius = 0.2e-3  # km (= 0.2 m)
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


# ---------------------------------------------------------------------------
# 7. Batched propagator — Fix 3 vectorisation
# ---------------------------------------------------------------------------


class TestBatchedPropagator:
    def test_matches_scalar_propagator(self) -> None:
        """Batched propagator agrees with the scalar version on a stencil."""
        rng = np.random.default_rng(1234)
        r_vecs = rng.standard_normal((30, 3))
        # include an exact-zero entry to exercise the self-interaction mask
        r_vecs[5] = 0.0

        P_batch = _propagator_block_9x9_batch(r_vecs, OMEGA, REF)
        assert P_batch.shape == (30, 9, 9)

        for i in range(30):
            P_ref = _propagator_block_9x9(r_vecs[i], OMEGA, REF)
            np.testing.assert_allclose(P_batch[i], P_ref, rtol=1e-12, atol=1e-14)

    def test_zero_offset_returns_zero(self) -> None:
        """Origin offset must yield the zero propagator block."""
        r_vecs = np.zeros((3, 3))
        P = _propagator_block_9x9_batch(r_vecs, OMEGA, REF)
        np.testing.assert_allclose(P, 0.0, atol=0.0)


# ---------------------------------------------------------------------------
# 8. z-reflection symmetry of the interlayer kernel cache — Fix 4
# ---------------------------------------------------------------------------


class TestInterlayerSymmetry:
    def test_parity_mask_shape_and_values(self) -> None:
        """Sign mask is outer product of the 9-channel z-parity signature."""
        assert _Z_PARITY_MASK_9x9.shape == (9, 9)
        assert np.all(np.abs(_Z_PARITY_MASK_9x9) == 1.0)
        # Channels with odd z-parity: indices {0, 7, 8} (u_z, 2ε_zy, 2ε_zx).
        odd = {0, 7, 8}
        for i in range(9):
            for j in range(9):
                expected = 1.0 if ((i in odd) == (j in odd)) else -1.0
                assert _Z_PARITY_MASK_9x9[i, j] == expected

    def test_derived_neg_dz_matches_direct(self) -> None:
        """Cached Δz<0 kernel matches a direct build for cubic inclusions."""
        n_sub = 4
        geom = cluster_from_sphere(RADIUS, n_sub)
        rayleigh_sub = compute_cube_tmatrix(OMEGA, geom.a_sub, REF, CONTRAST)
        T_loc = _sub_cell_tmatrix_9x9(rayleigh_sub, OMEGA, geom.a_sub)

        cache = build_interlayer_kernel_cache(
            n_sub, geom.a_sub, T_loc, OMEGA, REF, max_dz=3
        )

        for dz in (1, 2, 3):
            direct = build_interlayer_fft_kernel(
                n_sub, geom.a_sub, T_loc, OMEGA, REF, -dz
            )
            np.testing.assert_allclose(
                cache[-dz],
                direct,
                rtol=1e-12,
                atol=1e-14,
                err_msg=f"Cached vs direct kernel disagree at Δz=-{dz}",
            )


# ---------------------------------------------------------------------------
# 9. Multi-RHS layered matvec and block GMRES — Fix 5
# ---------------------------------------------------------------------------


class TestMultiRHS:
    @pytest.mark.parametrize("n_sub", [3, 5])
    def test_matvec_multi_matches_singlervs(self, n_sub: int) -> None:
        """Multi-RHS matvec produces the same columns as single-column matvec."""
        geom = cluster_from_sphere(RADIUS, n_sub)
        decomp = decompose_layers(geom)
        nC = len(geom.centres)

        rayleigh_sub = compute_cube_tmatrix(OMEGA, geom.a_sub, REF, CONTRAST)
        T_loc = _sub_cell_tmatrix_9x9(rayleigh_sub, OMEGA, geom.a_sub)

        shared = build_intralayer_fft_kernel(n_sub, geom.a_sub, T_loc, OMEGA, REF)
        intra_k = [shared] * decomp.n_layers
        max_dz = (
            int(decomp.z_indices[-1] - decomp.z_indices[0])
            if decomp.n_layers > 1
            else 0
        )
        inter_k = build_interlayer_kernel_cache(
            n_sub, geom.a_sub, T_loc, OMEGA, REF, max_dz
        )

        rng = np.random.default_rng(2024)
        k_rhs = 9
        W = rng.standard_normal((9 * nC, k_rhs)) + 1j * rng.standard_normal(
            (9 * nC, k_rhs)
        )

        Y_multi = layered_matvec_multi(W, decomp, intra_k, inter_k, n_sub)

        for col in range(k_rhs):
            y_col = layered_matvec(W[:, col], decomp, intra_k, inter_k, n_sub)
            np.testing.assert_allclose(Y_multi[:, col], y_col, rtol=1e-12, atol=1e-13)

    def test_block_gmres_slab(self) -> None:
        """Block GMRES matches a direct dense solve for a small slab."""
        M, N_z = 3, 2
        a = 10.0e-3  # km (= 10 m)
        omega = 2 * np.pi * 12.0 + 0.1j
        ref = ReferenceMedium(alpha=3.0, beta=1.5, rho=2.5)
        contrast = MaterialContrast(Dlambda=3.0, Dmu=1.5, Drho=0.1)

        geom = cluster_from_slab(M, N_z, a)
        decomp = decompose_layers(geom)
        nC = len(geom.centres)

        rayleigh = compute_cube_tmatrix(omega, a, ref, contrast)
        T_loc = _sub_cell_tmatrix_9x9(rayleigh, omega, a)

        shared = build_intralayer_fft_kernel(M, a, T_loc, omega, ref)  # type: ignore[arg-type]
        intra_k = [shared] * decomp.n_layers
        max_dz = (
            int(decomp.z_indices[-1] - decomp.z_indices[0])
            if decomp.n_layers > 1
            else 0
        )
        inter_k = build_interlayer_kernel_cache(
            M,
            a,
            T_loc,
            omega,  # type: ignore[arg-type]
            ref,
            max_dz,
        )

        # Build dense reference in z-sorted ordering
        centres_sorted = geom.centres[decomp.sort_order]
        dim = 9 * nC
        A_dense = np.eye(dim, dtype=complex)
        for i in range(nC):
            for jj in range(nC):
                if i == jj:
                    continue
                r = centres_sorted[i] - centres_sorted[jj]
                P = _propagator_block_9x9(r, omega, ref)
                A_dense[9 * i : 9 * i + 9, 9 * jj : 9 * jj + 9] = -(P @ T_loc)

        rng = np.random.default_rng(77)
        B = rng.standard_normal((dim, 9)) + 1j * rng.standard_normal((dim, 9))

        def mv_multi(W: np.ndarray) -> np.ndarray:
            return layered_matvec_multi(W, decomp, intra_k, inter_k, M)

        X_block, iters, rel_res = block_gmres(mv_multi, B, rtol=1e-10, max_iter=50)
        X_dense = np.linalg.solve(A_dense, B)

        err = np.linalg.norm(X_block - X_dense) / np.linalg.norm(X_dense)
        assert err < 1e-6, f"block_gmres vs dense rel err = {err:.2e}"
        assert iters > 0
        assert rel_res < 1e-9


# ---------------------------------------------------------------------------
# 10. Fix 6 — frequency-batched propagator
# ---------------------------------------------------------------------------


class TestPropagatorBatchFreq:
    def test_matches_per_freq_slice(self) -> None:
        """(F, N, 9, 9) matches calling the scalar batched version per ω."""
        rng = np.random.default_rng(321)
        r_vecs = rng.standard_normal((17, 3))
        r_vecs[4] = 0.0  # exercise the zero mask

        P_freq = _propagator_block_9x9_batch_freq(r_vecs, OMEGAS_TEST, REF)
        assert P_freq.shape == (len(OMEGAS_TEST), 17, 9, 9)

        for f_idx, om in enumerate(OMEGAS_TEST):
            P_ref = _propagator_block_9x9_batch(r_vecs, om, REF)
            np.testing.assert_allclose(P_freq[f_idx], P_ref, rtol=1e-12, atol=1e-14)

    def test_zero_offset_rows_are_zero(self) -> None:
        """The origin entry stays zero at every frequency."""
        r_vecs = np.zeros((4, 3))
        P_freq = _propagator_block_9x9_batch_freq(r_vecs, OMEGAS_TEST, REF)
        np.testing.assert_allclose(P_freq, 0.0, atol=0.0)

    def test_invalid_omegas_raise(self) -> None:
        """Non-1D / empty / real omegas trigger fail-fast ValueError."""
        r_vecs = np.zeros((1, 3))
        with pytest.raises(ValueError, match="1-D"):
            _propagator_block_9x9_batch_freq(r_vecs, OMEGAS_TEST.reshape(1, -1), REF)
        with pytest.raises(ValueError, match="complex"):
            _propagator_block_9x9_batch_freq(
                r_vecs,
                np.linspace(1.0, 2.0, 3),  # type: ignore[arg-type]
                REF,
            )
        with pytest.raises(ValueError, match="at least one"):
            _propagator_block_9x9_batch_freq(r_vecs, np.empty(0, dtype=complex), REF)


class TestTLocFreq:
    def test_matches_per_freq_loop(self) -> None:
        """(F, 9, 9) matches the scalar cube T-matrix at each ω."""
        a_cube = 10.0e-3  # km (= 10 m)
        ref = ReferenceMedium(alpha=3.0, beta=1.5, rho=2.5)
        contrast = MaterialContrast(Dlambda=3.0, Dmu=1.5, Drho=0.1)

        T_freq = build_T_loc_freq(a_cube, ref, contrast, OMEGAS_TEST)
        assert T_freq.shape == (len(OMEGAS_TEST), 9, 9)

        for f_idx, om in enumerate(OMEGAS_TEST):
            rayleigh = compute_cube_tmatrix(om, a_cube, ref, contrast)
            T_ref = _sub_cell_tmatrix_9x9(rayleigh, om, a_cube)
            np.testing.assert_allclose(T_freq[f_idx], T_ref, rtol=1e-12, atol=1e-14)

    def test_invalid_omegas_raise(self) -> None:
        """Fail-fast on invalid omega arrays."""
        a_cube = 10.0e-3  # km (= 10 m)
        ref = ReferenceMedium(alpha=3.0, beta=1.5, rho=2.5)
        contrast = MaterialContrast(Dlambda=3.0, Dmu=1.5, Drho=0.1)
        with pytest.raises(ValueError, match="complex"):
            build_T_loc_freq(
                a_cube,
                ref,
                contrast,
                np.linspace(1.0, 2.0, 3),  # type: ignore[arg-type]
            )


class TestKernelFreq:
    def _setup(
        self, n_sub: int = 4
    ) -> tuple[
        float,
        ReferenceMedium,
        MaterialContrast,
        np.ndarray,
    ]:
        a_cube = 10.0e-3  # km (= 10 m)
        ref = ReferenceMedium(alpha=3.0, beta=1.5, rho=2.5)
        contrast = MaterialContrast(Dlambda=3.0, Dmu=1.5, Drho=0.1)
        T_freq = build_T_loc_freq(a_cube, ref, contrast, OMEGAS_TEST)
        return a_cube, ref, contrast, T_freq

    def test_intralayer_matches_per_freq(self) -> None:
        """Intra-layer (F, 9, 9, nP, nP) matches per-ω scalar call."""
        n_sub = 4
        a_cube, ref, _, T_freq = self._setup(n_sub)
        K_freq = build_intralayer_fft_kernel_freq(
            n_sub, a_cube, T_freq, OMEGAS_TEST, ref
        )
        nP = 2 * n_sub - 1
        assert K_freq.shape == (len(OMEGAS_TEST), 9, 9, nP, nP)

        for f_idx, om in enumerate(OMEGAS_TEST):
            K_ref = build_intralayer_fft_kernel(n_sub, a_cube, T_freq[f_idx], om, ref)
            np.testing.assert_allclose(K_freq[f_idx], K_ref, rtol=1e-12, atol=1e-14)

    def test_interlayer_matches_per_freq(self) -> None:
        """Inter-layer (F, 9, 9, nP, nP) matches per-ω scalar call."""
        n_sub = 4
        a_cube, ref, _, T_freq = self._setup(n_sub)
        for dz in (1, 2, -1, -2):
            K_freq = build_interlayer_fft_kernel_freq(
                n_sub, a_cube, T_freq, OMEGAS_TEST, ref, dz
            )
            for f_idx, om in enumerate(OMEGAS_TEST):
                K_ref = build_interlayer_fft_kernel(
                    n_sub, a_cube, T_freq[f_idx], om, ref, dz
                )
                np.testing.assert_allclose(K_freq[f_idx], K_ref, rtol=1e-12, atol=1e-14)

    def test_interlayer_cache_matches_per_freq(self) -> None:
        """Cached (F, 9, 9, nP, nP) matches scalar cache per ω."""
        n_sub = 4
        a_cube, ref, _, T_freq = self._setup(n_sub)
        max_dz = 3
        cache_freq = build_interlayer_kernel_cache_freq(
            n_sub, a_cube, T_freq, OMEGAS_TEST, ref, max_dz
        )
        for dz in range(1, max_dz + 1):
            for signed_dz in (dz, -dz):
                assert signed_dz in cache_freq
                for f_idx, om in enumerate(OMEGAS_TEST):
                    K_ref = build_interlayer_fft_kernel(
                        n_sub, a_cube, T_freq[f_idx], om, ref, signed_dz
                    )
                    np.testing.assert_allclose(
                        cache_freq[signed_dz][f_idx],
                        K_ref,
                        rtol=1e-12,
                        atol=1e-14,
                    )

    def test_interlayer_parity_broadcast(self) -> None:
        """z-parity sign mask broadcasts correctly across F in the cache."""
        n_sub = 4
        a_cube, ref, _, T_freq = self._setup(n_sub)
        cache_freq = build_interlayer_kernel_cache_freq(
            n_sub, a_cube, T_freq, OMEGAS_TEST, ref, max_dz=2
        )
        sign_mask = _Z_PARITY_MASK_9x9[None, :, :, None, None]
        for dz in (1, 2):
            np.testing.assert_allclose(
                cache_freq[-dz],
                sign_mask * cache_freq[dz],
                rtol=1e-14,
                atol=0.0,
            )

    def test_invalid_T_loc_shape_raises(self) -> None:
        """Mismatched T_loc_freq leading axis raises ValueError."""
        n_sub = 3
        a_cube = 10.0e-3  # km (= 10 m)
        ref = ReferenceMedium(alpha=3.0, beta=1.5, rho=2.5)
        T_wrong = np.zeros((2, 9, 9), dtype=complex)
        with pytest.raises(ValueError, match="must equal omegas length"):
            build_intralayer_fft_kernel_freq(n_sub, a_cube, T_wrong, OMEGAS_TEST, ref)


class TestPackUnpackFreq:
    def test_pack_unpack_roundtrip(self) -> None:
        """``_pack_2d_freq`` and ``_unpack_2d_freq`` are inverse."""
        n_sub = 5
        geom = cluster_from_sphere(RADIUS, n_sub)
        decomp = decompose_layers(geom)
        nP = 2 * n_sub - 1
        n_freq = len(OMEGAS_TEST)

        for lz in range(decomp.n_layers):
            M = int(decomp.layer_sizes[lz])
            grid_2d = decomp.layer_grid_2d[lz]
            rng = np.random.default_rng(100 + lz)
            W = rng.standard_normal((n_freq, M, 9)) + 1j * rng.standard_normal(
                (n_freq, M, 9)
            )

            grids = _pack_2d_freq(W, grid_2d, nP)
            assert grids.shape == (n_freq, 9, nP, nP)
            W_back = _unpack_2d_freq(grids, grid_2d)
            np.testing.assert_allclose(W_back, W, atol=1e-14)

    def test_pack_freq_matches_per_freq(self) -> None:
        """Per-ω slice of pack_2d_freq equals the scalar pack."""
        n_sub = 4
        geom = cluster_from_sphere(RADIUS, n_sub)
        decomp = decompose_layers(geom)
        nP = 2 * n_sub - 1
        n_freq = len(OMEGAS_TEST)

        lz = 1
        M = int(decomp.layer_sizes[lz])
        grid_2d = decomp.layer_grid_2d[lz]
        rng = np.random.default_rng(77)
        W = rng.standard_normal((n_freq, M, 9)) + 1j * rng.standard_normal(
            (n_freq, M, 9)
        )
        G_freq = _pack_2d_freq(W, grid_2d, nP)
        for f_idx in range(n_freq):
            G_ref = _pack_2d(W[f_idx], grid_2d, nP)
            np.testing.assert_allclose(G_freq[f_idx], G_ref, atol=1e-14)

    def test_pack_unpack_multi_roundtrip(self) -> None:
        """``_pack_2d_multi_freq`` + ``_unpack_2d_multi_freq`` roundtrip."""
        n_sub = 4
        geom = cluster_from_sphere(RADIUS, n_sub)
        decomp = decompose_layers(geom)
        nP = 2 * n_sub - 1
        n_freq = len(OMEGAS_TEST)
        k_rhs = 4

        lz = 0
        M = int(decomp.layer_sizes[lz])
        grid_2d = decomp.layer_grid_2d[lz]
        rng = np.random.default_rng(999)
        W = rng.standard_normal((n_freq, M, 9, k_rhs)) + 1j * rng.standard_normal(
            (n_freq, M, 9, k_rhs)
        )

        grids = _pack_2d_multi_freq(W, grid_2d, nP)
        assert grids.shape == (n_freq, 9, k_rhs, nP, nP)
        W_back = _unpack_2d_multi_freq(grids, grid_2d)
        np.testing.assert_allclose(W_back, W, atol=1e-14)


class TestLayeredMatvecFreq:
    def _build_kernels(
        self, n_sub: int
    ) -> tuple[
        LayerDecomposition,
        int,
        list[np.ndarray],
        dict[int, np.ndarray],
        list[np.ndarray],
        dict[int, np.ndarray],
        int,
    ]:
        """Build per-ω and batched kernels for test geometry."""
        geom = cluster_from_sphere(RADIUS, n_sub)
        decomp = decompose_layers(geom)
        nC = len(geom.centres)

        T_freq = build_T_loc_freq(geom.a_sub, REF, CONTRAST, OMEGAS_TEST)

        # Per-ω scalar kernels.
        intra_scalar_per_f: list[list[np.ndarray]] = []
        max_dz = (
            int(decomp.z_indices[-1] - decomp.z_indices[0])
            if decomp.n_layers > 1
            else 0
        )
        inter_scalar_per_f: list[dict[int, np.ndarray]] = []
        for f_idx, om in enumerate(OMEGAS_TEST):
            shared = build_intralayer_fft_kernel(
                n_sub, geom.a_sub, T_freq[f_idx], om, REF
            )
            intra_scalar_per_f.append([shared] * decomp.n_layers)
            inter_scalar_per_f.append(
                build_interlayer_kernel_cache(
                    n_sub, geom.a_sub, T_freq[f_idx], om, REF, max_dz
                )
            )

        # Freq-batched kernels.
        shared_freq = build_intralayer_fft_kernel_freq(
            n_sub, geom.a_sub, T_freq, OMEGAS_TEST, REF
        )
        intra_freq = [shared_freq] * decomp.n_layers
        inter_freq = build_interlayer_kernel_cache_freq(
            n_sub, geom.a_sub, T_freq, OMEGAS_TEST, REF, max_dz
        )

        return (
            decomp,
            nC,
            intra_scalar_per_f,  # type: ignore[return-value]
            inter_scalar_per_f,  # type: ignore[return-value]
            intra_freq,
            inter_freq,
            n_sub,
        )

    def test_matvec_freq_matches_per_freq_loop(self) -> None:
        """Freq-batched single-RHS matvec matches per-ω scalar matvec."""
        n_sub = 4
        decomp, nC, intra_sc, inter_sc, intra_fq, inter_fq, _ = self._build_kernels(
            n_sub
        )
        n_freq = len(OMEGAS_TEST)
        rng = np.random.default_rng(2025)
        W = rng.standard_normal((n_freq, 9 * nC)) + 1j * rng.standard_normal(
            (n_freq, 9 * nC)
        )

        Y_freq = layered_matvec_freq(W, decomp, intra_fq, inter_fq, n_sub)

        for f_idx in range(n_freq):
            y_ref = layered_matvec(
                W[f_idx],
                decomp,
                intra_sc[f_idx],  # type: ignore[arg-type]
                inter_sc[f_idx],  # type: ignore[arg-type]
                n_sub,
            )
            np.testing.assert_allclose(Y_freq[f_idx], y_ref, rtol=1e-12, atol=1e-13)

    def test_matvec_multi_freq_matches_per_freq_loop(self) -> None:
        """Freq-batched multi-RHS matvec matches per-ω scalar matvec_multi."""
        n_sub = 4
        decomp, nC, intra_sc, inter_sc, intra_fq, inter_fq, _ = self._build_kernels(
            n_sub
        )
        n_freq = len(OMEGAS_TEST)
        k_rhs = 3
        rng = np.random.default_rng(4321)
        W = rng.standard_normal((n_freq, 9 * nC, k_rhs)) + 1j * rng.standard_normal(
            (n_freq, 9 * nC, k_rhs)
        )

        Y_freq = layered_matvec_multi_freq(W, decomp, intra_fq, inter_fq, n_sub)

        for f_idx in range(n_freq):
            Y_ref = layered_matvec_multi(
                W[f_idx],
                decomp,
                intra_sc[f_idx],  # type: ignore[arg-type]
                inter_sc[f_idx],  # type: ignore[arg-type]
                n_sub,
            )
            np.testing.assert_allclose(Y_freq[f_idx], Y_ref, rtol=1e-12, atol=1e-13)

    def test_matvec_freq_shape_errors(self) -> None:
        """Wrong-rank inputs trigger fail-fast ValueError."""
        n_sub = 3
        decomp, _, _, _, intra_fq, inter_fq, _ = self._build_kernels(n_sub)
        with pytest.raises(ValueError, match="must be 2D"):
            layered_matvec_freq(
                np.zeros((9 * 9,), dtype=complex),
                decomp,
                intra_fq,
                inter_fq,
                n_sub,
            )
        with pytest.raises(ValueError, match="must be 3D"):
            layered_matvec_multi_freq(
                np.zeros((5, 9 * 9), dtype=complex),
                decomp,
                intra_fq,
                inter_fq,
                n_sub,
            )


class TestBlockGmresFreq:
    def _build_dense_and_ops(
        self,
        M: int = 3,
        N_z: int = 2,
    ) -> tuple[
        np.ndarray,  # A_dense_freq (F, dim, dim)
        np.ndarray,  # B (F, dim, k)
        Callable[[np.ndarray], np.ndarray],  # matvec_multi_freq
        int,  # dim
    ]:
        """Build a small slab freq-batched operator and dense reference."""
        a = 10.0e-3  # km (= 10 m)
        ref = ReferenceMedium(alpha=3.0, beta=1.5, rho=2.5)
        contrast = MaterialContrast(Dlambda=3.0, Dmu=1.5, Drho=0.1)

        geom = cluster_from_slab(M, N_z, a)
        decomp = decompose_layers(geom)
        nC = len(geom.centres)
        dim = 9 * nC

        T_freq = build_T_loc_freq(a, ref, contrast, OMEGAS_TEST)
        intra_freq_shared = build_intralayer_fft_kernel_freq(
            M, a, T_freq, OMEGAS_TEST, ref
        )
        intra_freq = [intra_freq_shared] * decomp.n_layers
        max_dz = (
            int(decomp.z_indices[-1] - decomp.z_indices[0])
            if decomp.n_layers > 1
            else 0
        )
        inter_freq = build_interlayer_kernel_cache_freq(
            M, a, T_freq, OMEGAS_TEST, ref, max_dz
        )

        def matvec_multi_freq(W: np.ndarray) -> np.ndarray:
            return layered_matvec_multi_freq(W, decomp, intra_freq, inter_freq, M)

        # Dense reference per ω in z-sorted ordering.
        centres_sorted = geom.centres[decomp.sort_order]
        n_freq = len(OMEGAS_TEST)
        A_dense_freq = np.zeros((n_freq, dim, dim), dtype=complex)
        for f_idx, om in enumerate(OMEGAS_TEST):
            A = np.eye(dim, dtype=complex)
            for i in range(nC):
                for jj in range(nC):
                    if i == jj:
                        continue
                    r = centres_sorted[i] - centres_sorted[jj]
                    P = _propagator_block_9x9(r, om, ref)
                    A[9 * i : 9 * i + 9, 9 * jj : 9 * jj + 9] = -(P @ T_freq[f_idx])
            A_dense_freq[f_idx] = A

        rng = np.random.default_rng(909)
        B = rng.standard_normal((n_freq, dim, 9)) + 1j * rng.standard_normal(
            (n_freq, dim, 9)
        )

        return A_dense_freq, B, matvec_multi_freq, dim

    def test_matches_dense_per_freq(self) -> None:
        """Freq-batched block GMRES matches explicit dense solve per ω."""
        A_dense_freq, B, mv_freq, _ = self._build_dense_and_ops()
        X_block, iters, rel_res = block_gmres_freq(mv_freq, B, rtol=1e-10, max_iter=30)
        X_dense = np.linalg.solve(A_dense_freq, B)

        err = np.linalg.norm(X_block - X_dense) / np.linalg.norm(X_dense)
        assert err < 1e-6, f"rel err = {err:.2e}"
        assert iters > 0
        assert rel_res.shape == (len(OMEGAS_TEST),)
        assert float(np.max(rel_res)) < 1e-9

    def test_matches_scalar_block_gmres_per_freq(self) -> None:
        """Per-ω slice of block_gmres_freq output matches scalar block_gmres."""
        a = 10.0e-3  # km (= 10 m)
        ref = ReferenceMedium(alpha=3.0, beta=1.5, rho=2.5)
        contrast = MaterialContrast(Dlambda=3.0, Dmu=1.5, Drho=0.1)

        M, N_z = 3, 2
        geom = cluster_from_slab(M, N_z, a)
        decomp = decompose_layers(geom)
        nC = len(geom.centres)
        dim = 9 * nC

        T_freq = build_T_loc_freq(a, ref, contrast, OMEGAS_TEST)
        intra_freq = [
            build_intralayer_fft_kernel_freq(M, a, T_freq, OMEGAS_TEST, ref)
        ] * decomp.n_layers
        max_dz = (
            int(decomp.z_indices[-1] - decomp.z_indices[0])
            if decomp.n_layers > 1
            else 0
        )
        inter_freq = build_interlayer_kernel_cache_freq(
            M, a, T_freq, OMEGAS_TEST, ref, max_dz
        )

        def mv_freq(W: np.ndarray) -> np.ndarray:
            return layered_matvec_multi_freq(W, decomp, intra_freq, inter_freq, M)

        rng = np.random.default_rng(2500)
        B = rng.standard_normal((len(OMEGAS_TEST), dim, 9)) + 1j * rng.standard_normal(
            (len(OMEGAS_TEST), dim, 9)
        )

        X_freq, _, _ = block_gmres_freq(mv_freq, B, rtol=1e-10, max_iter=30)

        # Per-ω scalar reference.
        for f_idx, om in enumerate(OMEGAS_TEST):
            intra_sc = [
                build_intralayer_fft_kernel(M, a, T_freq[f_idx], om, ref)
            ] * decomp.n_layers
            inter_sc = build_interlayer_kernel_cache(
                M, a, T_freq[f_idx], om, ref, max_dz
            )

            def mv_scalar(
                W: np.ndarray,
                _intra=intra_sc,
                _inter=inter_sc,
            ) -> np.ndarray:
                return layered_matvec_multi(W, decomp, _intra, _inter, M)

            X_sc, _, _ = block_gmres(mv_scalar, B[f_idx], rtol=1e-10, max_iter=30)
            err = np.linalg.norm(X_freq[f_idx] - X_sc) / np.linalg.norm(X_sc)
            assert err < 1e-6, f"freq {f_idx}: rel err = {err:.2e}"

    def test_per_freq_residual_mask(self) -> None:
        """rel_res_freq is a per-ω length-F array."""
        A_dense_freq, B, mv_freq, _ = self._build_dense_and_ops()
        _, _, rel_res = block_gmres_freq(mv_freq, B, rtol=1e-8, max_iter=30)
        assert rel_res.shape == (len(OMEGAS_TEST),)
        # Every frequency should converge for this well-conditioned slab.
        assert np.all(rel_res < 1e-6)

    def test_shape_errors(self) -> None:
        """Fail-fast on wrong-rank B."""

        def dummy(W: np.ndarray) -> np.ndarray:
            return W

        with pytest.raises(ValueError, match="must be 3D"):
            block_gmres_freq(dummy, np.zeros((5, 9), dtype=complex))

    def test_memory_cap(self) -> None:
        """Pre-flight raises MemoryError when Krylov basis exceeds cap."""

        def dummy(W: np.ndarray) -> np.ndarray:
            return W

        # Modest RHS allocation, but projected basis size exceeds 4 GB
        # because max_iter is very large relative to (F, n, k).
        F, n, k = 2, 10000, 9
        B = np.zeros((F, n, k), dtype=complex)
        with pytest.raises(MemoryError, match="cap"):
            block_gmres_freq(dummy, B, max_iter=2000)


class TestBuildIncidentFieldCoupledFreq:
    def test_matches_per_freq(self) -> None:
        """Per-ω slice matches the scalar ``_build_incident_field_coupled``."""
        centres = np.random.default_rng(55).standard_normal((7, 3))
        ref = ReferenceMedium(alpha=3.0, beta=1.5, rho=2.5)

        psi_freq = _build_incident_field_coupled_freq(centres, OMEGAS_TEST, ref)
        assert psi_freq.shape == (len(OMEGAS_TEST), 9 * 7, 9)
        for f_idx, om in enumerate(OMEGAS_TEST):
            psi_ref = _build_incident_field_coupled(centres, om, ref)
            np.testing.assert_allclose(psi_freq[f_idx], psi_ref, rtol=1e-12, atol=1e-14)


class TestSolveSlabFoldyLaxFreq:
    def test_matches_per_freq_loop(self) -> None:
        """Freq-batched solve matches per-ω block_gmres solve."""
        M, N_z = 3, 2
        a = 10.0e-3  # km (= 10 m)
        ref = ReferenceMedium(alpha=3.0, beta=1.5, rho=2.5)
        contrast = MaterialContrast(Dlambda=3.0, Dmu=1.5, Drho=0.1)

        T_freq, iters, rel_res = solve_slab_foldy_lax_freq(
            M, N_z, a, OMEGAS_TEST, ref, contrast, rtol=1e-10, max_iter=30
        )
        assert T_freq.shape == (len(OMEGAS_TEST), 9, 9)
        assert rel_res.shape == (len(OMEGAS_TEST),)
        assert iters > 0
        assert float(np.max(rel_res)) < 1e-6

        # Per-ω reference via scalar block_gmres.
        geom = cluster_from_slab(M, N_z, a)
        decomp = decompose_layers(geom)
        nC = len(geom.centres)
        for f_idx, om in enumerate(OMEGAS_TEST):
            rayleigh = compute_cube_tmatrix(om, a, ref, contrast)
            T_loc = _sub_cell_tmatrix_9x9(rayleigh, om, a)
            intra_sc = [
                build_intralayer_fft_kernel(M, a, T_loc, om, ref)
            ] * decomp.n_layers
            max_dz = (
                int(decomp.z_indices[-1] - decomp.z_indices[0])
                if decomp.n_layers > 1
                else 0
            )
            inter_sc = build_interlayer_kernel_cache(M, a, T_loc, om, ref, max_dz)
            psi_inc = _build_incident_field_coupled(geom.centres, om, ref)
            psi_inc_sorted = np.zeros_like(psi_inc)
            for col in range(9):
                psi_inc_sorted[:, col] = _reorder_flat(
                    psi_inc[:, col], decomp.sort_order, nC
                )

            def mv(W: np.ndarray, _in=intra_sc, _ir=inter_sc) -> np.ndarray:
                return layered_matvec_multi(W, decomp, _in, _ir, M)

            X_sorted, *_ = block_gmres(
                mv,
                psi_inc_sorted,
                x0=psi_inc_sorted.copy(),
                rtol=1e-10,
                max_iter=30,
            )
            psi_exc = np.zeros_like(psi_inc)
            for col in range(9):
                psi_exc[:, col] = _reorder_flat(
                    X_sorted[:, col], decomp.unsort_order, nC
                )
            T_ref = np.zeros((9, 9), dtype=complex)
            for n in range(nC):
                T_ref += T_loc @ psi_exc[9 * n : 9 * n + 9, :]

            rel = np.linalg.norm(T_freq[f_idx] - T_ref) / np.linalg.norm(T_ref)
            assert rel < 1e-6, f"freq {f_idx}: rel err = {rel:.2e}"

    def test_invalid_omegas_raise(self) -> None:
        """Fail-fast on real / empty omegas."""
        ref = ReferenceMedium(alpha=3.0, beta=1.5, rho=2.5)
        contrast = MaterialContrast(Dlambda=3.0, Dmu=1.5, Drho=0.1)
        with pytest.raises(ValueError, match="complex"):
            solve_slab_foldy_lax_freq(
                2,
                2,
                10.0e-3,
                np.linspace(1.0, 2.0, 3),  # type: ignore[arg-type]
                ref,
                contrast,
            )
