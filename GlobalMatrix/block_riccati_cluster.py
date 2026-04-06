"""Block-preconditioned layered Foldy-Lax solver for voxelized sphere.

Groups cubes by depth (z-index) and exploits the Toeplitz structure
within each layer for efficient 2D FFT convolution.  The intra-layer
solve serves as a block-diagonal preconditioner for the outer GMRES,
which only needs to handle spectrally-convergent inter-layer residuals.

Physical insight:
    - Intra-layer (Δz=0): G₀ spectral kernel ~1/kH (divergent) —
      must solve in spatial domain.
    - Inter-layer (Δz≠0): G₀ picks up exp(-κ|Δz|) decay — spectrally
      convergent and well-behaved.

Architecture:
    Outer GMRES with ``layered_matvec`` (full (I − G·T) via 2D FFT
    per layer pair) preconditioned by ``block_preconditioner`` (solve
    each layer's intra-layer system independently via inner GMRES +
    2D FFT convolution).
"""

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator, gmres

_MS_ROOT = Path("/Users/tod/Desktop/MultipleScatteringCalculations")
if str(_MS_ROOT) not in sys.path:
    sys.path.insert(0, str(_MS_ROOT))

from cubic_scattering import (  # noqa: E402
    MaterialContrast,
    ReferenceMedium,
    compute_cube_tmatrix,
)
from cubic_scattering.resonance_tmatrix import (  # noqa: E402
    _build_incident_field_coupled,
    _propagator_block_9x9,
    _sub_cell_tmatrix_9x9,
)
from cubic_scattering.sphere_scattering_fft import _build_grid_index_map  # noqa: E402


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ClusterGeometry:
    """Geometry of a voxelized sphere on a regular grid.

    Args:
        n_sub: Bounding grid size per edge.
        a_sub: Cube half-width (m).
        grid_idx: Integer grid indices, shape (N_total, 3) as (iz, ix, iy).
        centres: Physical centre coordinates, shape (N_total, 3).
    """

    n_sub: int
    a_sub: float
    grid_idx: NDArray[np.intp]
    centres: NDArray[np.floating]


@dataclass
class LayerDecomposition:
    """Layer-by-layer decomposition of a cluster.

    Args:
        n_layers: Number of distinct z-layers.
        z_indices: Sorted unique z grid indices, shape (n_layers,).
        layer_sizes: Number of cubes per layer, shape (n_layers,).
        layer_slices: Index slices into the flat (sorted-by-z) vector.
        layer_grid_2d: Per-layer (M_z, 2) arrays of (ix, iy) grid indices.
        sort_order: Permutation to go from original to sorted-by-z ordering.
        unsort_order: Inverse permutation (sorted-by-z back to original).
    """

    n_layers: int
    z_indices: NDArray[np.intp]
    layer_sizes: NDArray[np.intp]
    layer_slices: list[slice]
    layer_grid_2d: list[NDArray[np.intp]]
    sort_order: NDArray[np.intp]
    unsort_order: NDArray[np.intp]


@dataclass
class BlockRiccatiResult:
    """Result from block-preconditioned layered Foldy-Lax solver.

    Args:
        T3x3: 3×3 effective displacement T-matrix.
        T_comp_9x9: Full 9×9 composite T-matrix.
        centres: Sub-cell centre coordinates, shape (N, 3).
        n_sub: Bounding grid size per edge.
        n_cells: Number of cells inside sphere.
        a_sub: Sub-cell half-width (m).
        psi_exc: Exciting field solution, shape (9*N, 9).
        omega: Angular frequency.
        radius: Sphere radius (m).
        ref: Background medium.
        contrast: Material contrast.
        gmres_iters: Number of outer GMRES iterations.
        n_layers: Number of z-layers.
    """

    T3x3: NDArray[np.complexfloating]
    T_comp_9x9: NDArray[np.complexfloating]
    centres: NDArray[np.floating]
    n_sub: int
    n_cells: int
    a_sub: float
    psi_exc: NDArray[np.complexfloating]
    omega: float
    radius: float
    ref: ReferenceMedium
    contrast: MaterialContrast
    gmres_iters: int
    n_layers: int


# ---------------------------------------------------------------------------
# 1. Geometry and layer decomposition
# ---------------------------------------------------------------------------


def cluster_from_sphere(radius: float, n_sub: int) -> ClusterGeometry:
    """Build cluster geometry from a voxelized sphere.

    Reuses ``_build_grid_index_map`` from ``sphere_scattering_fft.py``
    and reindexes to (iz, ix, iy) ordering for layer decomposition.

    Args:
        radius: Sphere radius (m).
        n_sub: Bounding grid size per edge.

    Returns:
        ClusterGeometry with grid indices in (iz, ix, iy) ordering.
    """
    grid_idx, centres, a_sub = _build_grid_index_map(radius, n_sub)
    # _build_grid_index_map returns (i0, i1, i2) — already Cartesian grid
    # Convention: treat axis 0 as z, axis 1 as x, axis 2 as y
    return ClusterGeometry(
        n_sub=n_sub,
        a_sub=a_sub,
        grid_idx=grid_idx,
        centres=centres,
    )


def cluster_from_slab(M: int, N_z: int, a: float) -> ClusterGeometry:
    """Build cluster geometry for a rectangular slab of space-filling cubes.

    Creates an M × M × N_z grid of cubes, centred at the origin.
    Axis convention: (z, x, y) — axis 0 is depth.

    Args:
        M: Lateral grid size (cubes per edge in x and y).
        N_z: Number of z-layers.
        a: Cube half-width (m). Cube side d = 2a (space-filling).

    Returns:
        ClusterGeometry with n_sub = M.
    """
    d = 2.0 * a
    grid_indices = []
    centres_list = []
    for iz in range(N_z):
        for ix in range(M):
            for iy in range(M):
                grid_indices.append([iz, ix, iy])
                pos = np.array(
                    [
                        (iz - (N_z - 1) / 2.0) * d,
                        (ix - (M - 1) / 2.0) * d,
                        (iy - (M - 1) / 2.0) * d,
                    ]
                )
                centres_list.append(pos)

    return ClusterGeometry(
        n_sub=M,
        a_sub=a,
        grid_idx=np.array(grid_indices, dtype=np.intp),
        centres=np.array(centres_list, dtype=float),
    )


def decompose_layers(geometry: ClusterGeometry) -> LayerDecomposition:
    """Decompose cluster into layers by z-index.

    Sorts cubes by z-index, computes per-layer slices and 2D grid indices.

    Args:
        geometry: Cluster geometry.

    Returns:
        LayerDecomposition with all layer metadata.
    """
    gi = geometry.grid_idx  # (N, 3) as (iz, ix, iy)

    # Sort by z-index (stable sort preserves relative order within layer)
    sort_order = np.argsort(gi[:, 0], kind="stable")
    unsort_order = np.argsort(sort_order)

    sorted_gi = gi[sort_order]
    z_vals = sorted_gi[:, 0]

    z_unique, counts = np.unique(z_vals, return_counts=True)
    n_layers = len(z_unique)

    layer_slices: list[slice] = []
    layer_grid_2d: list[NDArray[np.intp]] = []
    offset = 0
    for i in range(n_layers):
        s = slice(offset, offset + counts[i])
        layer_slices.append(s)
        # 2D grid indices for this layer: (ix, iy)
        layer_grid_2d.append(sorted_gi[s, 1:3].copy())
        offset += counts[i]

    return LayerDecomposition(
        n_layers=n_layers,
        z_indices=z_unique.astype(np.intp),
        layer_sizes=counts.astype(np.intp),
        layer_slices=layer_slices,
        layer_grid_2d=layer_grid_2d,
        sort_order=sort_order,
        unsort_order=unsort_order,
    )


# ---------------------------------------------------------------------------
# 2. Intra-layer 2D FFT kernel (Δz = 0)
# ---------------------------------------------------------------------------


def build_intralayer_fft_kernel(
    n_sub: int,
    a_sub: float,
    T_loc: NDArray[np.complexfloating],
    omega: float,
    ref: ReferenceMedium,
) -> NDArray[np.complexfloating]:
    """Build 2D FFT kernel for intra-layer convolution (Δz=0).

    For all (Δx, Δy) in [-(n_sub-1), +(n_sub-1)]², evaluates
    ``-P_9x9([0, Δx·d, Δy·d]) @ T_loc`` with circular embedding
    on (2·n_sub - 1)² grid, then 2D FFTs each of 81 components.

    Note: This kernel depends on both the propagator (free-space here)
    and T_loc.  For a uniform sphere in free space, T_loc is the same
    for all cubes and the free-space propagator is translationally
    invariant, so one kernel suffices.  For a general plane-layer stack
    (non-uniform T_loc or layered-medium Green's function), build a
    separate kernel per layer.

    Args:
        n_sub: Grid size per edge.
        a_sub: Cube half-width (m).
        T_loc: Local 9×9 T-matrix.
        omega: Angular frequency (rad/s).
        ref: Background medium.

    Returns:
        kernel_hat: shape (9, 9, nP, nP), complex FFT of kernel.
    """
    dd = 2.0 * a_sub
    nP = 2 * n_sub - 1

    kernel = np.zeros((9, 9, nP, nP), dtype=complex)

    for dx in range(-(n_sub - 1), n_sub):
        for dy in range(-(n_sub - 1), n_sub):
            if dx == 0 and dy == 0:
                continue
            r_vec = np.array([0.0, dx * dd, dy * dd])
            P_block = _propagator_block_9x9(r_vec, omega, ref)
            block = -(P_block @ T_loc)
            ix = dx % nP
            iy = dy % nP
            kernel[:, :, ix, iy] = block

    kernel_hat = np.zeros_like(kernel)
    for i in range(9):
        for j in range(9):
            kernel_hat[i, j] = np.fft.fft2(kernel[i, j])

    return kernel_hat


# ---------------------------------------------------------------------------
# 3. Inter-layer 2D FFT kernels (Δz ≠ 0)
# ---------------------------------------------------------------------------


def build_interlayer_fft_kernel(
    n_sub: int,
    a_sub: float,
    T_loc: NDArray[np.complexfloating],
    omega: float,
    ref: ReferenceMedium,
    dz_cubes: int,
) -> NDArray[np.complexfloating]:
    """Build 2D FFT kernel for inter-layer convolution at given Δz.

    For all (Δx, Δy) in [-(n_sub-1), +(n_sub-1)]², evaluates
    ``-P_9x9([dz_cubes·d, Δx·d, Δy·d]) @ T_loc`` with circular
    embedding on (2·n_sub - 1)² grid, then 2D FFTs.

    Args:
        n_sub: Grid size per edge.
        a_sub: Cube half-width (m).
        T_loc: Local 9×9 T-matrix.
        omega: Angular frequency (rad/s).
        ref: Background medium.
        dz_cubes: Signed layer separation in grid units (Δz ≠ 0).

    Returns:
        kernel_hat: shape (9, 9, nP, nP), complex FFT of kernel.
    """
    dd = 2.0 * a_sub
    nP = 2 * n_sub - 1

    kernel = np.zeros((9, 9, nP, nP), dtype=complex)

    for dx in range(-(n_sub - 1), n_sub):
        for dy in range(-(n_sub - 1), n_sub):
            r_vec = np.array([dz_cubes * dd, dx * dd, dy * dd])
            P_block = _propagator_block_9x9(r_vec, omega, ref)
            block = -(P_block @ T_loc)
            ix = dx % nP
            iy = dy % nP
            kernel[:, :, ix, iy] = block

    kernel_hat = np.zeros_like(kernel)
    for i in range(9):
        for j in range(9):
            kernel_hat[i, j] = np.fft.fft2(kernel[i, j])

    return kernel_hat


def build_interlayer_kernel_cache(
    n_sub: int,
    a_sub: float,
    T_loc: NDArray[np.complexfloating],
    omega: float,
    ref: ReferenceMedium,
    max_dz: int,
) -> dict[int, NDArray[np.complexfloating]]:
    """Build cache of inter-layer kernels for |Δz| = 1..max_dz.

    Exploits z-reflection symmetry: the 9×9 block structure gives
    - G, S blocks (3×3 disp, 6×6 strain): even in Δz
    - C, H blocks (3×6, 6×3 coupling): odd in Δz

    Only computes for Δz > 0, derives Δz < 0 by sign flip on C/H.

    Args:
        n_sub: Grid size per edge.
        a_sub: Cube half-width (m).
        T_loc: Local 9×9 T-matrix.
        omega: Angular frequency (rad/s).
        ref: Background medium.
        max_dz: Maximum |Δz| in grid units.

    Returns:
        Dict mapping signed Δz → FFT kernel, shape (9, 9, nP, nP).
    """
    cache: dict[int, NDArray[np.complexfloating]] = {}

    for dz in range(1, max_dz + 1):
        kernel_pos = build_interlayer_fft_kernel(n_sub, a_sub, T_loc, omega, ref, dz)
        cache[dz] = kernel_pos

        # Derive negative Δz via z-reflection symmetry
        # P = [[G, C], [H, S]] where G(3x3), C(3x6), H(6x3), S(6x6)
        # Under z-flip: G → G, S → S, C → -C, H → -H
        # kernel stores -P@T_loc, so sign relationships are:
        # kernel_neg = flip(kernel_pos) on the C/H sub-blocks of P
        # But since kernel = -P@T, the relationship depends on the
        # full 9x9 structure. Compute directly for correctness.
        kernel_neg = build_interlayer_fft_kernel(n_sub, a_sub, T_loc, omega, ref, -dz)
        cache[-dz] = kernel_neg

    return cache


# ---------------------------------------------------------------------------
# 2D pack/unpack helpers
# ---------------------------------------------------------------------------


def _pack_2d(
    w_block: NDArray[np.complexfloating],
    grid_2d: NDArray[np.intp],
    nP: int,
) -> NDArray[np.complexfloating]:
    """Pack (M, 9) block onto (9, nP, nP) grid.

    Args:
        w_block: Data for layer cubes, shape (M, 9).
        grid_2d: 2D grid indices (ix, iy), shape (M, 2).
        nP: Padded 2D grid size (2*n_sub - 1).

    Returns:
        grids: shape (9, nP, nP), zero-padded.
    """
    grids = np.zeros((9, nP, nP), dtype=complex)
    for c in range(9):
        grids[c, grid_2d[:, 0], grid_2d[:, 1]] = w_block[:, c]
    return grids


def _unpack_2d(
    grids: NDArray[np.complexfloating],
    grid_2d: NDArray[np.intp],
    M: int,
) -> NDArray[np.complexfloating]:
    """Unpack (9, nP, nP) grid to (M, 9) block.

    Args:
        grids: Grid data, shape (9, nP, nP).
        grid_2d: 2D grid indices (ix, iy), shape (M, 2).
        M: Number of cubes in this layer.

    Returns:
        w_block: shape (M, 9).
    """
    w_block = np.zeros((M, 9), dtype=complex)
    for c in range(9):
        w_block[:, c] = grids[c, grid_2d[:, 0], grid_2d[:, 1]]
    return w_block


def _apply_2d_fft_kernel(
    w_block: NDArray[np.complexfloating],
    kernel_hat: NDArray[np.complexfloating],
    grid_2d: NDArray[np.intp],
    nP: int,
    M: int,
) -> NDArray[np.complexfloating]:
    """Apply 2D FFT convolution kernel to a layer's data.

    Computes kernel * w via: pack → FFT → pointwise multiply → IFFT → unpack.

    Args:
        w_block: Input data, shape (M, 9).
        kernel_hat: FFT of kernel, shape (9, 9, nP, nP).
        grid_2d: 2D grid indices, shape (M, 2).
        nP: Padded 2D grid size.
        M: Number of cubes in layer.

    Returns:
        Result, shape (M, 9).
    """
    grids = _pack_2d(w_block, grid_2d, nP)

    # FFT each component
    w_hat = np.zeros_like(grids)
    for c in range(9):
        w_hat[c] = np.fft.fft2(grids[c])

    # Pointwise 9×9 multiply in frequency domain
    y_hat = np.zeros_like(w_hat)
    for i in range(9):
        for j in range(9):
            y_hat[i] += kernel_hat[i, j] * w_hat[j]

    # IFFT and unpack
    y_grids = np.zeros_like(y_hat)
    for c in range(9):
        y_grids[c] = np.fft.ifft2(y_hat[c])

    return _unpack_2d(y_grids, grid_2d, M)


# ---------------------------------------------------------------------------
# 4. Full layered matvec: (I - G·T)·w
# ---------------------------------------------------------------------------


def layered_matvec(
    w_flat: NDArray[np.complexfloating],
    decomp: LayerDecomposition,
    intralayer_kernels: list[NDArray[np.complexfloating]],
    interlayer_kernels: dict[int, NDArray[np.complexfloating]],
    n_sub: int,
) -> NDArray[np.complexfloating]:
    """Compute (I - G·T)·w via layered 2D FFT convolution.

    The input/output vectors are in sorted-by-z ordering.

    Args:
        w_flat: Input vector, shape (9*N_total,), in z-sorted order.
        decomp: Layer decomposition metadata.
        intralayer_kernels: Per-layer FFT kernels for Δz=0.
            ``intralayer_kernels[lz]`` has shape (9, 9, nP, nP).
            For uniform T_loc (e.g. voxelized sphere), all entries
            may reference the same array.
        interlayer_kernels: Dict mapping signed Δz → FFT kernel.
        n_sub: Grid size per edge.

    Returns:
        Result vector, shape (9*N_total,), in z-sorted order.
    """
    nP = 2 * n_sub - 1
    result = w_flat.copy()  # Start with identity term

    # Extract per-layer blocks
    layer_blocks = []
    for lz in range(decomp.n_layers):
        s = decomp.layer_slices[lz]
        M = decomp.layer_sizes[lz]
        block = w_flat[9 * s.start : 9 * s.stop].reshape(M, 9)
        layer_blocks.append(block)

    # Accumulate -(P·T)·w contributions via 2D FFT convolution
    for lr in range(decomp.n_layers):
        sr = decomp.layer_slices[lr]
        Mr = decomp.layer_sizes[lr]
        grid_r = decomp.layer_grid_2d[lr]
        accum = np.zeros((Mr, 9), dtype=complex)

        for ls in range(decomp.n_layers):
            grid_s = decomp.layer_grid_2d[ls]
            Ms = decomp.layer_sizes[ls]

            dz = int(decomp.z_indices[lr] - decomp.z_indices[ls])

            if dz == 0:
                # Intra-layer: use layer-specific kernel
                accum += _apply_2d_fft_kernel_cross(
                    layer_blocks[ls],
                    intralayer_kernels[ls],
                    grid_s,
                    grid_r,
                    nP,
                    Ms,
                    Mr,
                )
            else:
                # Inter-layer: use Δz-specific kernel
                if dz not in interlayer_kernels:
                    continue
                accum += _apply_2d_fft_kernel_cross(
                    layer_blocks[ls],
                    interlayer_kernels[dz],
                    grid_s,
                    grid_r,
                    nP,
                    Ms,
                    Mr,
                )

        # Add convolution result (kernel already has -P@T sign)
        result[9 * sr.start : 9 * sr.stop] += accum.ravel()

    return result


def _apply_2d_fft_kernel_cross(
    w_block_src: NDArray[np.complexfloating],
    kernel_hat: NDArray[np.complexfloating],
    grid_src: NDArray[np.intp],
    grid_rcv: NDArray[np.intp],
    nP: int,
    M_src: int,
    M_rcv: int,
) -> NDArray[np.complexfloating]:
    """Apply 2D FFT convolution with source and receiver on different grids.

    Packs source data, convolves via FFT, unpacks at receiver grid points.

    Args:
        w_block_src: Source data, shape (M_src, 9).
        kernel_hat: FFT of kernel, shape (9, 9, nP, nP).
        grid_src: Source 2D grid indices, shape (M_src, 2).
        grid_rcv: Receiver 2D grid indices, shape (M_rcv, 2).
        nP: Padded 2D grid size.
        M_src: Number of source cubes.
        M_rcv: Number of receiver cubes.

    Returns:
        Result at receiver points, shape (M_rcv, 9).
    """
    grids = _pack_2d(w_block_src, grid_src, nP)

    w_hat = np.zeros_like(grids)
    for c in range(9):
        w_hat[c] = np.fft.fft2(grids[c])

    y_hat = np.zeros_like(w_hat)
    for i in range(9):
        for j in range(9):
            y_hat[i] += kernel_hat[i, j] * w_hat[j]

    y_grids = np.zeros((9, nP, nP), dtype=complex)
    for c in range(9):
        y_grids[c] = np.fft.ifft2(y_hat[c])

    return _unpack_2d(y_grids, grid_rcv, M_rcv)


# ---------------------------------------------------------------------------
# 5. Block-diagonal preconditioner
# ---------------------------------------------------------------------------


def block_preconditioner(
    r_flat: NDArray[np.complexfloating],
    decomp: LayerDecomposition,
    intralayer_kernels: list[NDArray[np.complexfloating]],
    n_sub: int,
    inner_tol: float = 1e-4,
    inner_maxiter: int = 50,
) -> NDArray[np.complexfloating]:
    """Block-Jacobi preconditioner: solve each layer independently.

    For each layer z, solves ``(I - G_intra·T) · x_z = r_z`` using
    GMRES + 2D FFT matvec.

    Args:
        r_flat: Residual vector, shape (9*N_total,), in z-sorted order.
        decomp: Layer decomposition metadata.
        intralayer_kernels: Per-layer FFT kernels for Δz=0.
            ``intralayer_kernels[lz]`` has shape (9, 9, nP, nP).
        n_sub: Grid size per edge.
        inner_tol: GMRES tolerance for inner solves.
        inner_maxiter: Max iterations for inner solves.

    Returns:
        Preconditioned vector, shape (9*N_total,).
    """
    nP = 2 * n_sub - 1
    result = np.zeros_like(r_flat)

    for lz in range(decomp.n_layers):
        s = decomp.layer_slices[lz]
        M = decomp.layer_sizes[lz]
        grid_2d = decomp.layer_grid_2d[lz]
        dim = 9 * M
        kernel_lz = intralayer_kernels[lz]

        rhs = r_flat[9 * s.start : 9 * s.stop]

        if M == 0:
            continue

        def layer_matvec(w: NDArray, _g=grid_2d, _M=M, _k=kernel_lz) -> NDArray:
            w_block = w.reshape(_M, 9)
            conv = _apply_2d_fft_kernel(w_block, _k, _g, nP, _M)
            return w + conv.ravel()

        A_layer = LinearOperator((dim, dim), matvec=layer_matvec, dtype=complex)

        x, info = gmres(
            A_layer, rhs, x0=rhs.copy(), rtol=inner_tol, maxiter=inner_maxiter
        )
        if info != 0:
            warnings.warn(
                f"Inner GMRES for layer {lz} did not converge (info={info})",
                UserWarning,
                stacklevel=2,
            )
        result[9 * s.start : 9 * s.stop] = x

    return result


# ---------------------------------------------------------------------------
# 6. Top-level solver
# ---------------------------------------------------------------------------


def compute_cluster_scattering(
    omega: float,
    radius: float,
    ref: ReferenceMedium,
    contrast: MaterialContrast,
    n_sub: int,
    k_hat: NDArray | None = None,
    wave_type: str = "S",
    preconditioner: str = "block_jacobi",
    gmres_tol: float = 1e-8,
    gmres_maxiter: int = 200,
    inner_tol: float = 1e-4,
    inner_maxiter: int = 50,
) -> BlockRiccatiResult:
    """Compute sphere T-matrix via block-preconditioned layered Foldy-Lax.

    Args:
        omega: Angular frequency (rad/s).
        radius: Sphere radius (m).
        ref: Background medium.
        contrast: Material contrasts.
        n_sub: Number of sub-cells per edge of bounding cube.
        k_hat: Unit incident direction (default z-hat).
        wave_type: 'S' or 'P'.
        preconditioner: 'block_jacobi' or 'none'.
        gmres_tol: Relative tolerance for outer GMRES.
        gmres_maxiter: Maximum outer GMRES iterations.
        inner_tol: Tolerance for inner (preconditioner) GMRES.
        inner_maxiter: Max iterations for inner GMRES.

    Returns:
        BlockRiccatiResult with composite T-matrix.
    """
    # 1. Build geometry and decompose layers
    geom = cluster_from_sphere(radius, n_sub)
    decomp = decompose_layers(geom)
    nC = len(geom.centres)

    # 2. Build T_loc
    rayleigh_sub = compute_cube_tmatrix(omega, geom.a_sub, ref, contrast)
    T_loc = _sub_cell_tmatrix_9x9(rayleigh_sub, omega, geom.a_sub)

    # 3. Build per-layer intra-layer kernels + inter-layer kernels
    # For a uniform sphere in free space, all layers share the same kernel
    # (same T_loc, translationally invariant propagator).  Build once, reuse.
    shared_intralayer_kernel = build_intralayer_fft_kernel(
        n_sub, geom.a_sub, T_loc, omega, ref
    )
    intralayer_kernels = [shared_intralayer_kernel] * decomp.n_layers

    max_dz = (
        int(decomp.z_indices[-1] - decomp.z_indices[0]) if decomp.n_layers > 1 else 0
    )
    interlayer_kernels = build_interlayer_kernel_cache(
        n_sub, geom.a_sub, T_loc, omega, ref, max_dz
    )

    # 4. Build incident field
    psi_inc = _build_incident_field_coupled(
        geom.centres, omega, ref, k_hat=k_hat, wave_type=wave_type
    )

    # 5. Set up layered matvec as LinearOperator
    dim = 9 * nC

    def full_matvec(w: NDArray) -> NDArray:
        # Convert from original ordering to sorted-by-z
        w_sorted = _reorder_flat(w, decomp.sort_order, nC)
        y_sorted = layered_matvec(
            w_sorted, decomp, intralayer_kernels, interlayer_kernels, n_sub
        )
        # Convert back to original ordering
        return _reorder_flat(y_sorted, decomp.unsort_order, nC)

    A_op = LinearOperator((dim, dim), matvec=full_matvec, dtype=complex)

    # 6. Set up preconditioner
    if preconditioner == "block_jacobi":

        def precond_matvec(r: NDArray) -> NDArray:
            r_sorted = _reorder_flat(r, decomp.sort_order, nC)
            x_sorted = block_preconditioner(
                r_sorted, decomp, intralayer_kernels, n_sub, inner_tol, inner_maxiter
            )
            return _reorder_flat(x_sorted, decomp.unsort_order, nC)

        M_op = LinearOperator((dim, dim), matvec=precond_matvec, dtype=complex)
    else:
        M_op = None

    # 7. Solve column by column via preconditioned GMRES
    psi_exc = np.zeros((dim, 9), dtype=complex)
    total_iters = 0
    for col in range(9):
        rhs = psi_inc[:, col]
        x0 = rhs.copy()

        # Track iterations via callback
        col_iters = [0]

        def _count_iter(xk, _ci=col_iters):
            _ci[0] += 1

        solution, info = gmres(
            A_op,
            rhs,
            x0=x0,
            rtol=gmres_tol,
            maxiter=gmres_maxiter,
            M=M_op,
            callback=_count_iter,
            callback_type="x",
        )
        if info != 0:
            warnings.warn(
                f"Outer GMRES did not converge for column {col} (info={info})",
                UserWarning,
                stacklevel=2,
            )
        psi_exc[:, col] = solution
        total_iters += col_iters[0]

    # 8. Extract composite T-matrix
    T_comp = np.zeros((9, 9), dtype=complex)
    for n in range(nC):
        T_comp += T_loc @ psi_exc[9 * n : 9 * n + 9, :]

    T3x3 = T_comp[:3, :3].copy()

    return BlockRiccatiResult(
        T3x3=T3x3,
        T_comp_9x9=T_comp,
        centres=geom.centres,
        n_sub=n_sub,
        n_cells=nC,
        a_sub=geom.a_sub,
        psi_exc=psi_exc,
        omega=omega,
        radius=radius,
        ref=ref,
        contrast=contrast,
        gmres_iters=total_iters,
        n_layers=decomp.n_layers,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reorder_flat(
    v: NDArray[np.complexfloating],
    perm: NDArray[np.intp],
    nC: int,
) -> NDArray[np.complexfloating]:
    """Reorder a flat 9*nC vector using a permutation on cube indices.

    Args:
        v: Flat vector, shape (9*nC,).
        perm: Permutation of cube indices, shape (nC,).
        nC: Number of cubes.

    Returns:
        Reordered flat vector, shape (9*nC,).
    """
    blocks = v.reshape(nC, 9)
    return blocks[perm].ravel()
