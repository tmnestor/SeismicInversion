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
    VOIGT_PAIRS,
    _build_incident_field_coupled,
    _sub_cell_tmatrix_9x9,
)
from cubic_scattering.sphere_scattering_fft import _build_grid_index_map  # noqa: E402


# ---------------------------------------------------------------------------
# 9-channel z-parity signature
# ---------------------------------------------------------------------------
#
# Under z-reflection ``z → -z`` each 9-component channel picks up a sign
# equal to ``(-1)^{number of z-indices in the tensor component}``.  With the
# channel ordering ``[u_z, u_x, u_y, ε_zz, ε_xx, ε_yy, 2ε_xy, 2ε_zy, 2ε_zx]``
# (matching ``VOIGT_PAIRS`` which uses Cartesian index 0 for z), the
# z-parity signature is:
Z_PARITY_SIGNS = np.array([-1, 1, 1, 1, 1, 1, 1, -1, -1], dtype=np.int8)
# The (9, 9) sign mask for the relationship
#   ``kernel(-Δz)[i, j] = η_i η_j · kernel(+Δz)[i, j]``
# is the outer product.  This identity holds whenever the local T-matrix
# commutes with ``diag(η)``, which is the case for cubic-symmetry inclusions
# (block-diagonal T with equal shear stiffness entries).
_Z_PARITY_MASK_9x9 = np.outer(Z_PARITY_SIGNS, Z_PARITY_SIGNS).astype(float)


# ---------------------------------------------------------------------------
# Batched (N, 3) → (N, 9, 9) propagator — Fix 3 vectorisation
# ---------------------------------------------------------------------------


def _radial_functions_batch(
    r: NDArray[np.floating], kP: float, kS: float
) -> tuple[
    NDArray[np.complexfloating],
    NDArray[np.complexfloating],
    NDArray[np.complexfloating],
    NDArray[np.complexfloating],
    NDArray[np.complexfloating],
    NDArray[np.complexfloating],
]:
    """Batched version of ``_radial_functions`` over distance array ``r``.

    Args:
        r: Distance array, shape (N,), strictly positive.
        kP: P-wave wavenumber (may be complex if ω has imaginary part).
        kS: S-wave wavenumber.

    Returns:
        ``(φ, ψ, φ', ψ', φ'', ψ'')`` — each array has shape (N,), complex.
    """
    expP = np.exp(1j * kP * r)
    expS = np.exp(1j * kS * r)
    exps = {kP: expP, kS: expS}

    phi_terms: list[tuple[complex, float, int]] = [
        (kS**2, kS, -1),
        (1j * kS, kS, -2),
        (-1.0, kS, -3),
        (-1j * kP, kP, -2),
        (1.0, kP, -3),
    ]
    psi_terms: list[tuple[complex, float, int]] = [
        (-(kS**2), kS, -1),
        (-3j * kS, kS, -2),
        (3.0, kS, -3),
        (kP**2, kP, -1),
        (3j * kP, kP, -2),
        (-3.0, kP, -3),
    ]

    def _accumulate(
        terms: list[tuple[complex, float, int]],
    ) -> tuple[
        NDArray[np.complexfloating],
        NDArray[np.complexfloating],
        NDArray[np.complexfloating],
    ]:
        f = np.zeros_like(r, dtype=complex)
        f_p = np.zeros_like(r, dtype=complex)
        f_pp = np.zeros_like(r, dtype=complex)
        for c, k, p in terms:
            e = exps[k]
            ikr = 1j * k * r
            rp = r**p
            f += c * e * rp
            f_p += c * e * r ** (p - 1) * (ikr + p)
            f_pp += c * e * r ** (p - 2) * (ikr**2 + 2 * p * ikr + p * (p - 1))
        return f, f_p, f_pp

    phi, phi_p, phi_pp = _accumulate(phi_terms)
    psi, psi_p, psi_pp = _accumulate(psi_terms)
    return phi, psi, phi_p, psi_p, phi_pp, psi_pp


def _elastodynamic_greens_deriv_batch(
    r_vecs: NDArray[np.floating],
    omega: float,
    ref: ReferenceMedium,
) -> tuple[
    NDArray[np.complexfloating],
    NDArray[np.complexfloating],
    NDArray[np.complexfloating],
]:
    """Batched Green's tensor and first/second derivatives.

    Vectorises :func:`elastodynamic_greens_deriv` over a stack of ``N``
    relative position vectors.  Entries with ``‖r_vec‖ < 1e-14`` are
    returned as zeros (self-interaction handled in the local T-matrix).

    Args:
        r_vecs: Relative position vectors, shape (N, 3).
        omega: Angular frequency (rad/s).
        ref: Background medium.

    Returns:
        ``(G, Gd, Gdd)`` with shapes ``(N, 3, 3)``, ``(N, 3, 3, 3)`` and
        ``(N, 3, 3, 3, 3)``, all complex.
    """
    r_vecs = np.asarray(r_vecs, dtype=float)
    if r_vecs.ndim != 2 or r_vecs.shape[1] != 3:
        msg = f"r_vecs must have shape (N, 3); got {r_vecs.shape}"
        raise ValueError(msg)
    n_total = r_vecs.shape[0]

    G_out = np.zeros((n_total, 3, 3), dtype=complex)
    Gd_out = np.zeros((n_total, 3, 3, 3), dtype=complex)
    Gdd_out = np.zeros((n_total, 3, 3, 3, 3), dtype=complex)

    r = np.linalg.norm(r_vecs, axis=-1)
    nonzero = r > 1.0e-14
    if not np.any(nonzero):
        return G_out, Gd_out, Gdd_out

    idx = np.nonzero(nonzero)[0]
    rv = r_vecs[idx]
    rn = r[idx]
    g = rv / rn[:, None]  # (M, 3)
    inv_r = 1.0 / rn
    inv_r2 = inv_r**2

    kP: float = omega / ref.alpha
    kS: float = omega / ref.beta
    prefac = 1.0 / (4.0 * np.pi * ref.rho * omega**2)
    phi, psi, phi_p, psi_p, phi_pp, psi_pp = _radial_functions_batch(rn, kP, kS)

    delta = np.eye(3)

    # ----- G_{ij}  (M, 3, 3) -----
    gg = np.einsum("mi,mj->mij", g, g)
    G_sub = prefac * (phi[:, None, None] * delta[None, :, :] + psi[:, None, None] * gg)
    G_out[idx] = G_sub

    # ----- G_{ij,k}  (M, 3, 3, 3) -----
    c1 = phi_p
    c2 = psi_p - 2.0 * psi * inv_r
    c3 = psi * inv_r
    Gd_sub = prefac * (
        c1[:, None, None, None] * np.einsum("mk,ij->mijk", g, delta)
        + c2[:, None, None, None] * np.einsum("mi,mj,mk->mijk", g, g, g)
        + c3[:, None, None, None]
        * (np.einsum("ik,mj->mijk", delta, g) + np.einsum("jk,mi->mijk", delta, g))
    )
    Gd_out[idx] = Gd_sub

    # ----- G_{ij,kl}  (M, 3, 3, 3, 3)  — 7 tensor structures -----
    t1 = phi_p * inv_r
    t2 = phi_pp - phi_p * inv_r
    t3 = psi_p * inv_r - 2.0 * psi * inv_r2
    t4 = psi * inv_r2
    t7 = psi_pp - 5.0 * psi_p * inv_r + 8.0 * psi * inv_r2

    dd_ij_dd_kl = np.einsum("ij,kl->ijkl", delta, delta)[None]  # (1, 3, 3, 3, 3)
    dd_ik_dd_jl = np.einsum("ik,jl->ijkl", delta, delta)[None]
    dd_jk_dd_il = np.einsum("jk,il->ijkl", delta, delta)[None]

    Gdd_sub = prefac * (
        t1[:, None, None, None, None] * dd_ij_dd_kl
        + t2[:, None, None, None, None] * np.einsum("ij,mk,ml->mijkl", delta, g, g)
        + t3[:, None, None, None, None] * np.einsum("mi,mj,kl->mijkl", g, g, delta)
        + t4[:, None, None, None, None] * (dd_ik_dd_jl + dd_jk_dd_il)
        + t3[:, None, None, None, None]
        * (
            np.einsum("il,mj,mk->mijkl", delta, g, g)
            + np.einsum("jl,mi,mk->mijkl", delta, g, g)
        )
        + t3[:, None, None, None, None]
        * (
            np.einsum("ik,mj,ml->mijkl", delta, g, g)
            + np.einsum("jk,mi,ml->mijkl", delta, g, g)
        )
        + t7[:, None, None, None, None] * np.einsum("mi,mj,mk,ml->mijkl", g, g, g, g)
    )
    Gdd_out[idx] = Gdd_sub

    return G_out, Gd_out, Gdd_out


def _voigt_contract_batch(
    Gd: NDArray[np.complexfloating],
    Gdd: NDArray[np.complexfloating],
) -> tuple[
    NDArray[np.complexfloating],
    NDArray[np.complexfloating],
    NDArray[np.complexfloating],
]:
    """Batched Voigt contraction of ``Gd`` and ``Gdd``.

    Args:
        Gd: First derivative G_{ij,k}, shape (N, 3, 3, 3).
        Gdd: Second derivative G_{ij,kl}, shape (N, 3, 3, 3, 3).

    Returns:
        ``(C, H, S)`` with shapes ``(N, 3, 6)``, ``(N, 6, 3)``, ``(N, 6, 6)``.
    """
    n_total = Gd.shape[0]
    C = np.zeros((n_total, 3, 6), dtype=complex)
    H = np.zeros((n_total, 6, 3), dtype=complex)
    S = np.zeros((n_total, 6, 6), dtype=complex)

    for alpha, (p, q) in enumerate(VOIGT_PAIRS):
        if p == q:
            C[:, :, alpha] = Gd[:, :, p, p]
            H[:, alpha, :] = Gd[:, p, :, p]
        else:
            C[:, :, alpha] = Gd[:, :, p, q] + Gd[:, :, q, p]
            H[:, alpha, :] = Gd[:, p, :, q] + Gd[:, q, :, p]

        for beta, (m, n) in enumerate(VOIGT_PAIRS):
            if p == q and m == n:
                S[:, alpha, beta] = Gdd[:, p, m, m, p]
            elif p == q and m != n:
                S[:, alpha, beta] = Gdd[:, p, m, n, p] + Gdd[:, p, n, m, p]
            elif p != q and m == n:
                S[:, alpha, beta] = Gdd[:, p, m, m, q] + Gdd[:, q, m, m, p]
            else:
                S[:, alpha, beta] = (
                    Gdd[:, p, m, n, q]
                    + Gdd[:, p, n, m, q]
                    + Gdd[:, q, m, n, p]
                    + Gdd[:, q, n, m, p]
                )

    # Engineering-strain factor on off-diagonal Voigt columns
    C[:, :, 3:] *= 0.5
    S[:, :, 3:] *= 0.5
    return C, H, S


def _propagator_block_9x9_batch(
    r_vecs: NDArray[np.floating],
    omega: float,
    ref: ReferenceMedium,
) -> NDArray[np.complexfloating]:
    """Batched 9x9 inter-sub-cell propagator.

    Fix 3 vectorisation of :func:`_propagator_block_9x9`.  Takes a stack
    of relative position vectors and returns a stack of propagator blocks
    in one pass.  Entries with ``‖r_vec‖ < 1e-14`` are zero.

    Args:
        r_vecs: Relative position vectors, shape (N, 3).
        omega: Angular frequency (rad/s).
        ref: Background medium.

    Returns:
        Propagator stack, shape (N, 9, 9), complex.
    """
    G, Gd, Gdd = _elastodynamic_greens_deriv_batch(r_vecs, omega, ref)
    C, H, S = _voigt_contract_batch(Gd, Gdd)
    n_total = G.shape[0]
    P = np.zeros((n_total, 9, 9), dtype=complex)
    P[:, :3, :3] = G
    P[:, :3, 3:] = C
    P[:, 3:, :3] = H
    P[:, 3:, 3:] = S
    return P


# ---------------------------------------------------------------------------
# Fix 6 — Frequency axis: validation helper and batched propagator
# ---------------------------------------------------------------------------


def _validate_omegas(omegas: NDArray[np.complexfloating]) -> None:
    """Fail-fast validation of a frequency-sweep omega array.

    Required by all public ``*_freq`` entry points in the block Foldy-Lax
    pipeline.  Raises :class:`ValueError` with actionable messages.

    Args:
        omegas: Frequency array, must be 1-D complex with len > 0.

    Raises:
        ValueError: If ``omegas`` is not 1-D, not complex, or empty.
    """
    arr = np.asarray(omegas)
    if arr.ndim != 1:
        msg = (
            f"omegas must be 1-D, got ndim={arr.ndim}. "
            "Pass np.asarray(list_of_freqs).ravel() or a rank-1 array."
        )
        raise ValueError(msg)
    if arr.dtype.kind != "c":
        msg = (
            f"omegas must be complex dtype, got {arr.dtype}. "
            "Multi-frequency kernels require complex ω to handle damping; "
            "cast via omegas = np.asarray(omegas, dtype=complex) or add a "
            "small imaginary part (e.g. 1e-2j) for stability."
        )
        raise ValueError(msg)
    if arr.size == 0:
        msg = "omegas must contain at least one frequency; got empty array."
        raise ValueError(msg)


def _propagator_block_9x9_batch_freq(
    r_vecs: NDArray[np.floating],
    omegas: NDArray[np.complexfloating],
    ref: ReferenceMedium,
) -> NDArray[np.complexfloating]:
    """Frequency-batched 9x9 propagator stack.

    Fix 6 entry point: loops over ω internally, calling the existing
    :func:`_propagator_block_9x9_batch` at each frequency.  The ω loop is
    unavoidable (``exp(ikr)`` is nonlinear in ω) but the per-ω stencil
    and pre-allocation work is amortised in the caller.

    Args:
        r_vecs: Relative position vectors, shape (N, 3).
        omegas: Complex angular frequencies, shape (F,).
        ref: Background medium.

    Returns:
        Propagator stack, shape (F, N, 9, 9), complex.
    """
    _validate_omegas(omegas)
    r_vecs = np.asarray(r_vecs, dtype=float)
    n_total = r_vecs.shape[0]
    n_freq = omegas.shape[0]
    out = np.empty((n_freq, n_total, 9, 9), dtype=complex)
    for f_idx, om in enumerate(omegas):
        out[f_idx] = _propagator_block_9x9_batch(r_vecs, om, ref)
    return out


def build_T_loc_freq(
    a: float,
    ref: ReferenceMedium,
    contrast: MaterialContrast,
    omegas: NDArray[np.complexfloating],
) -> NDArray[np.complexfloating]:
    """Frequency-batched local 9×9 cube T-matrix.

    Loops ω internally, calling :func:`compute_cube_tmatrix` and
    :func:`_sub_cell_tmatrix_9x9`.  The Python-level loop is unavoidable
    because both routines embed transcendental dispersion through ω; the
    batch API exists purely to provide a uniform ``*_freq`` shape for
    downstream kernel builders.

    Args:
        a: Cube half-width (m).
        ref: Background medium.
        contrast: Material contrast of the cube.
        omegas: Complex angular frequencies, shape (F,).

    Returns:
        Stack of local T-matrices, shape (F, 9, 9), complex.
    """
    _validate_omegas(omegas)
    n_freq = omegas.shape[0]
    out = np.empty((n_freq, 9, 9), dtype=complex)
    for f_idx, om in enumerate(omegas):
        rayleigh = compute_cube_tmatrix(om, a, ref, contrast)
        out[f_idx] = _sub_cell_tmatrix_9x9(rayleigh, om, a)
    return out


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


def _stencil_offsets(
    n_sub: int, dz: int
) -> tuple[NDArray[np.floating], NDArray[np.intp], NDArray[np.intp]]:
    """Build all (dz·d, dx·d, dy·d) offsets for a (2n-1)×(2n-1) stencil.

    Args:
        n_sub: Grid size per edge.
        dz: Signed z-separation in grid units.

    Returns:
        ``(r_vecs, ix_flat, iy_flat)`` where ``r_vecs`` has shape
        ``(N, 3)`` and the index arrays give the circular embedding
        position for each offset.  Here ``N = (2n_sub-1)²``.
    """
    offsets = np.arange(-(n_sub - 1), n_sub)  # shape (2n-1,)
    dx_grid, dy_grid = np.meshgrid(offsets, offsets, indexing="ij")
    dx_flat = dx_grid.ravel()
    dy_flat = dy_grid.ravel()
    n_pts = dx_flat.size

    nP = 2 * n_sub - 1
    r_vecs = np.zeros((n_pts, 3), dtype=float)
    r_vecs[:, 0] = float(dz)
    r_vecs[:, 1] = dx_flat.astype(float)
    r_vecs[:, 2] = dy_flat.astype(float)
    # caller multiplies by ``dd`` — keep integers here for clarity
    ix_flat = (dx_flat % nP).astype(np.intp)
    iy_flat = (dy_flat % nP).astype(np.intp)
    return r_vecs, ix_flat, iy_flat


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

    The propagator stencil is evaluated in one batched call (Fix 3) and
    the per-channel FFTs are batched over the channel axes (Fix 1).

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

    # Build all (Δx, Δy) offsets for Δz=0 in one stencil call
    r_vecs, ix_flat, iy_flat = _stencil_offsets(n_sub, dz=0)
    r_vecs[:, 0] *= dd  # Δz = 0 already
    r_vecs[:, 1] *= dd
    r_vecs[:, 2] *= dd

    # Batched propagator (zeros at the origin offset by construction)
    P_batch = _propagator_block_9x9_batch(r_vecs, omega, ref)  # (N, 9, 9)
    # Contract each block with T_loc: -P @ T_loc, still (N, 9, 9)
    block_batch = -np.einsum("nij,jk->nik", P_batch, T_loc)

    kernel = np.zeros((9, 9, nP, nP), dtype=complex)
    kernel[:, :, ix_flat, iy_flat] = np.moveaxis(block_batch, 0, -1)

    # Batched 2D FFT over the last two axes of the (9, 9, nP, nP) tensor
    return np.fft.fft2(kernel, axes=(-2, -1))


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
    embedding on (2·n_sub - 1)² grid, then 2D FFTs.  Uses the
    Fix 3 batched propagator and the Fix 1 batched FFT.

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

    r_vecs, ix_flat, iy_flat = _stencil_offsets(n_sub, dz=dz_cubes)
    r_vecs[:, 0] *= dd
    r_vecs[:, 1] *= dd
    r_vecs[:, 2] *= dd

    P_batch = _propagator_block_9x9_batch(r_vecs, omega, ref)  # (N, 9, 9)
    block_batch = -np.einsum("nij,jk->nik", P_batch, T_loc)

    kernel = np.zeros((9, 9, nP, nP), dtype=complex)
    kernel[:, :, ix_flat, iy_flat] = np.moveaxis(block_batch, 0, -1)

    return np.fft.fft2(kernel, axes=(-2, -1))


def build_interlayer_kernel_cache(
    n_sub: int,
    a_sub: float,
    T_loc: NDArray[np.complexfloating],
    omega: float,
    ref: ReferenceMedium,
    max_dz: int,
) -> dict[int, NDArray[np.complexfloating]]:
    """Build cache of inter-layer kernels for |Δz| = 1..max_dz.

    Exploits z-reflection symmetry (Fix 4): under ``z → -z``, the 9×9
    propagator transforms as ``P(-Δz) = D P(Δz) D`` where
    ``D = diag(Z_PARITY_SIGNS)``.  When the local T-matrix commutes with
    ``D`` (the case for cubic-symmetry inclusions), the kernel satisfies

        kernel(-Δz)[i, j] = η_i η_j · kernel(+Δz)[i, j]

    entrywise (and the same identity holds in FFT space because the FFT
    is linear and acts only on the spatial axes).  This halves the kernel
    construction work: we compute ``+Δz`` directly and derive ``-Δz`` via
    the precomputed sign mask.

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
    sign_mask = _Z_PARITY_MASK_9x9[:, :, None, None]  # (9, 9, 1, 1)

    for dz in range(1, max_dz + 1):
        kernel_pos = build_interlayer_fft_kernel(n_sub, a_sub, T_loc, omega, ref, dz)
        cache[dz] = kernel_pos
        cache[-dz] = sign_mask * kernel_pos

    return cache


# ---------------------------------------------------------------------------
# Fix 6 — Frequency-batched kernel builders
# ---------------------------------------------------------------------------


def build_intralayer_fft_kernel_freq(
    n_sub: int,
    a_sub: float,
    T_loc_freq: NDArray[np.complexfloating],
    omegas: NDArray[np.complexfloating],
    ref: ReferenceMedium,
) -> NDArray[np.complexfloating]:
    """Frequency-batched intra-layer 2D FFT kernel (Δz=0).

    Precomputes the ``(Δx, Δy)`` stencil, flat indices and scaled
    ``r_vecs`` once, then fills the ``(F, 9, 9, nP, nP)`` spatial-domain
    kernel tensor via a Python loop over ω (one batched propagator call
    and one einsum per ω).  The final 2D FFT is a single batched call
    over the leading ``(F, 9, 9)`` axes (numpy's ``fft2`` accepts
    arbitrary batch axes ahead of ``axes=(-2, -1)``).

    Args:
        n_sub: Grid size per edge.
        a_sub: Cube half-width (m).
        T_loc_freq: Local 9×9 T-matrix stack, shape (F, 9, 9).
        omegas: Complex angular frequencies, shape (F,).
        ref: Background medium.

    Returns:
        kernel_hat_freq: shape (F, 9, 9, nP, nP), complex FFT of kernel.
    """
    _validate_omegas(omegas)
    if T_loc_freq.shape[0] != omegas.shape[0]:
        msg = (
            f"T_loc_freq leading axis {T_loc_freq.shape[0]} "
            f"must equal omegas length {omegas.shape[0]}"
        )
        raise ValueError(msg)

    dd = 2.0 * a_sub
    nP = 2 * n_sub - 1
    n_freq = omegas.shape[0]

    # Stencil is ω-independent — amortised outside the loop.
    r_vecs, ix_flat, iy_flat = _stencil_offsets(n_sub, dz=0)
    r_vecs[:, 0] *= dd  # Δz = 0 already
    r_vecs[:, 1] *= dd
    r_vecs[:, 2] *= dd

    kernel = np.zeros((n_freq, 9, 9, nP, nP), dtype=complex)
    for f_idx, om in enumerate(omegas):
        P_batch = _propagator_block_9x9_batch(r_vecs, om, ref)  # (N, 9, 9)
        block_batch = -np.einsum("nij,jk->nik", P_batch, T_loc_freq[f_idx])
        # Here the scalar f_idx + the (ix_flat, iy_flat) advanced indices are
        # separated by two basic slices, so numpy moves the combined advanced
        # axis to the front: destination shape is (N, 9, 9), matching
        # block_batch directly.
        kernel[f_idx, :, :, ix_flat, iy_flat] = block_batch

    # One batched 2D FFT over the last two axes for the full (F, 9, 9) stack.
    return np.fft.fft2(kernel, axes=(-2, -1))


def build_interlayer_fft_kernel_freq(
    n_sub: int,
    a_sub: float,
    T_loc_freq: NDArray[np.complexfloating],
    omegas: NDArray[np.complexfloating],
    ref: ReferenceMedium,
    dz_cubes: int,
) -> NDArray[np.complexfloating]:
    """Frequency-batched inter-layer 2D FFT kernel at given Δz.

    Same amortisation strategy as
    :func:`build_intralayer_fft_kernel_freq`: stencil computed once,
    ω loop fills the ``(F, 9, 9, nP, nP)`` spatial buffer, single batched
    2D FFT at the end.

    Args:
        n_sub: Grid size per edge.
        a_sub: Cube half-width (m).
        T_loc_freq: Local 9×9 T-matrix stack, shape (F, 9, 9).
        omegas: Complex angular frequencies, shape (F,).
        ref: Background medium.
        dz_cubes: Signed layer separation in grid units (Δz ≠ 0).

    Returns:
        kernel_hat_freq: shape (F, 9, 9, nP, nP), complex FFT of kernel.
    """
    _validate_omegas(omegas)
    if T_loc_freq.shape[0] != omegas.shape[0]:
        msg = (
            f"T_loc_freq leading axis {T_loc_freq.shape[0]} "
            f"must equal omegas length {omegas.shape[0]}"
        )
        raise ValueError(msg)

    dd = 2.0 * a_sub
    nP = 2 * n_sub - 1
    n_freq = omegas.shape[0]

    r_vecs, ix_flat, iy_flat = _stencil_offsets(n_sub, dz=dz_cubes)
    r_vecs[:, 0] *= dd
    r_vecs[:, 1] *= dd
    r_vecs[:, 2] *= dd

    kernel = np.zeros((n_freq, 9, 9, nP, nP), dtype=complex)
    for f_idx, om in enumerate(omegas):
        P_batch = _propagator_block_9x9_batch(r_vecs, om, ref)
        block_batch = -np.einsum("nij,jk->nik", P_batch, T_loc_freq[f_idx])
        kernel[f_idx, :, :, ix_flat, iy_flat] = block_batch

    return np.fft.fft2(kernel, axes=(-2, -1))


def build_interlayer_kernel_cache_freq(
    n_sub: int,
    a_sub: float,
    T_loc_freq: NDArray[np.complexfloating],
    omegas: NDArray[np.complexfloating],
    ref: ReferenceMedium,
    max_dz: int,
) -> dict[int, NDArray[np.complexfloating]]:
    """Frequency-batched cache of inter-layer kernels for ``|Δz| ≤ max_dz``.

    Mirrors :func:`build_interlayer_kernel_cache` but each cache entry
    is a ``(F, 9, 9, nP, nP)`` tensor.  The z-reflection parity identity
    still holds entrywise, and the ``_Z_PARITY_MASK_9x9`` mask broadcasts
    over the leading frequency axis without modification.

    Args:
        n_sub: Grid size per edge.
        a_sub: Cube half-width (m).
        T_loc_freq: Local 9×9 T-matrix stack, shape (F, 9, 9).
        omegas: Complex angular frequencies, shape (F,).
        ref: Background medium.
        max_dz: Maximum ``|Δz|`` in grid units.

    Returns:
        Dict mapping signed Δz → FFT kernel, shape (F, 9, 9, nP, nP).
    """
    _validate_omegas(omegas)
    cache: dict[int, NDArray[np.complexfloating]] = {}
    # Broadcast sign mask across leading F axis implicitly.
    sign_mask = _Z_PARITY_MASK_9x9[None, :, :, None, None]  # (1, 9, 9, 1, 1)

    for dz in range(1, max_dz + 1):
        kernel_pos = build_interlayer_fft_kernel_freq(
            n_sub, a_sub, T_loc_freq, omegas, ref, dz
        )
        cache[dz] = kernel_pos
        cache[-dz] = sign_mask * kernel_pos

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
    # Vectorised scatter over the 9 channels in one fancy-index write
    grids[:, grid_2d[:, 0], grid_2d[:, 1]] = w_block.T
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
    # Vectorised gather over the 9 channels — returns (9, M), transpose.
    return grids[:, grid_2d[:, 0], grid_2d[:, 1]].T


def _apply_2d_fft_kernel(
    w_block: NDArray[np.complexfloating],
    kernel_hat: NDArray[np.complexfloating],
    grid_2d: NDArray[np.intp],
    nP: int,
    M: int,
) -> NDArray[np.complexfloating]:
    """Apply 2D FFT convolution kernel to a layer's data.

    Computes kernel * w via: pack → batched FFT → einsum 9×9 mix → batched
    IFFT → unpack.  Fix 1 (batched FFT over channels) + Fix 2 (einsum in
    place of the explicit 9×9 channel loop).

    Args:
        w_block: Input data, shape (M, 9).
        kernel_hat: FFT of kernel, shape (9, 9, nP, nP).
        grid_2d: 2D grid indices, shape (M, 2).
        nP: Padded 2D grid size.
        M: Number of cubes in layer.

    Returns:
        Result, shape (M, 9).
    """
    grids = _pack_2d(w_block, grid_2d, nP)  # (9, nP, nP)
    w_hat = np.fft.fft2(grids, axes=(-2, -1))  # Fix 1 — one batched FFT
    # Fix 2 — one einsum instead of 81 broadcasted adds
    y_hat = np.einsum("ijxy,jxy->ixy", kernel_hat, w_hat)
    y_grids = np.fft.ifft2(y_hat, axes=(-2, -1))
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
    grids = _pack_2d(w_block_src, grid_src, nP)  # (9, nP, nP)
    w_hat = np.fft.fft2(grids, axes=(-2, -1))  # Fix 1
    y_hat = np.einsum("ijxy,jxy->ixy", kernel_hat, w_hat)  # Fix 2
    y_grids = np.fft.ifft2(y_hat, axes=(-2, -1))
    return _unpack_2d(y_grids, grid_rcv, M_rcv)


# ---------------------------------------------------------------------------
# 4b. Multi-RHS matvec: (I - G·T) · W with W a (9*N, k) block — Fix 5
# ---------------------------------------------------------------------------


def _pack_2d_multi(
    W_block: NDArray[np.complexfloating],
    grid_2d: NDArray[np.intp],
    nP: int,
) -> NDArray[np.complexfloating]:
    """Pack (M, 9, k) data onto (9, k, nP, nP) grid.

    Args:
        W_block: Multi-RHS data, shape (M, 9, k).
        grid_2d: 2D grid indices, shape (M, 2).
        nP: Padded 2D grid size.

    Returns:
        grids: shape (9, k, nP, nP), zero-padded.
    """
    k_rhs = W_block.shape[2]
    grids = np.zeros((9, k_rhs, nP, nP), dtype=complex)
    # Scatter: destination shape at advanced-index is (9, k, M);
    # source should be transposed to (9, k, M).
    grids[:, :, grid_2d[:, 0], grid_2d[:, 1]] = W_block.transpose(1, 2, 0)
    return grids


def _unpack_2d_multi(
    grids: NDArray[np.complexfloating],
    grid_2d: NDArray[np.intp],
) -> NDArray[np.complexfloating]:
    """Unpack (9, k, nP, nP) grid to (M, 9, k)."""
    return grids[:, :, grid_2d[:, 0], grid_2d[:, 1]].transpose(2, 0, 1)


# ---------------------------------------------------------------------------
# Fix 6 — Frequency-batched pack/unpack helpers
# ---------------------------------------------------------------------------


def _pack_2d_freq(
    W_block: NDArray[np.complexfloating],
    grid_2d: NDArray[np.intp],
    nP: int,
) -> NDArray[np.complexfloating]:
    """Frequency-batched pack of ``(F, M, 9)`` onto ``(F, 9, nP, nP)``.

    Args:
        W_block: Data for layer cubes, shape (F, M, 9).
        grid_2d: 2D grid indices (ix, iy), shape (M, 2).
        nP: Padded 2D grid size.

    Returns:
        grids: shape (F, 9, nP, nP), zero-padded.
    """
    n_freq = W_block.shape[0]
    grids = np.zeros((n_freq, 9, nP, nP), dtype=complex)
    # Advanced indices at (-2, -1) are contiguous; leading basic slices
    # keep F and channel axes in place so destination shape is (F, 9, M).
    grids[:, :, grid_2d[:, 0], grid_2d[:, 1]] = W_block.transpose(0, 2, 1)
    return grids


def _unpack_2d_freq(
    grids: NDArray[np.complexfloating],
    grid_2d: NDArray[np.intp],
) -> NDArray[np.complexfloating]:
    """Frequency-batched unpack of ``(F, 9, nP, nP)`` to ``(F, M, 9)``."""
    return grids[:, :, grid_2d[:, 0], grid_2d[:, 1]].transpose(0, 2, 1)


def _pack_2d_multi_freq(
    W_block: NDArray[np.complexfloating],
    grid_2d: NDArray[np.intp],
    nP: int,
) -> NDArray[np.complexfloating]:
    """Frequency-batched multi-RHS pack ``(F, M, 9, k) → (F, 9, k, nP, nP)``.

    Args:
        W_block: Multi-RHS data, shape (F, M, 9, k).
        grid_2d: 2D grid indices, shape (M, 2).
        nP: Padded 2D grid size.

    Returns:
        grids: shape (F, 9, k, nP, nP).
    """
    n_freq = W_block.shape[0]
    k_rhs = W_block.shape[3]
    grids = np.zeros((n_freq, 9, k_rhs, nP, nP), dtype=complex)
    # Destination shape at advanced index (F, 9, k, M); source is
    # W_block transposed to (F, 9, k, M).
    grids[:, :, :, grid_2d[:, 0], grid_2d[:, 1]] = W_block.transpose(0, 2, 3, 1)
    return grids


def _unpack_2d_multi_freq(
    grids: NDArray[np.complexfloating],
    grid_2d: NDArray[np.intp],
) -> NDArray[np.complexfloating]:
    """Frequency-batched multi-RHS unpack ``(F, 9, k, nP, nP) → (F, M, 9, k)``."""
    return grids[:, :, :, grid_2d[:, 0], grid_2d[:, 1]].transpose(0, 3, 1, 2)


def _apply_2d_fft_kernel_cross_freq(
    W_block_src: NDArray[np.complexfloating],
    kernel_hat_freq: NDArray[np.complexfloating],
    grid_src: NDArray[np.intp],
    grid_rcv: NDArray[np.intp],
    nP: int,
) -> NDArray[np.complexfloating]:
    """Frequency-batched single-RHS 2D FFT convolution.

    Applies a ``(F, 9, 9, nP, nP)`` kernel to a ``(F, M_src, 9)`` source
    block in one batched FFT + einsum, unpacking at the receiver grid.

    Args:
        W_block_src: Source data, shape (F, M_src, 9).
        kernel_hat_freq: FFT of kernel, shape (F, 9, 9, nP, nP).
        grid_src: Source 2D grid indices, shape (M_src, 2).
        grid_rcv: Receiver 2D grid indices, shape (M_rcv, 2).
        nP: Padded 2D grid size.

    Returns:
        Result at receiver points, shape (F, M_rcv, 9).
    """
    grids = _pack_2d_freq(W_block_src, grid_src, nP)  # (F, 9, nP, nP)
    w_hat = np.fft.fft2(grids, axes=(-2, -1))
    # Mixes the 9-channel axis per frequency; "f" axis broadcasts cleanly.
    y_hat = np.einsum("fijxy,fjxy->fixy", kernel_hat_freq, w_hat)
    y_grids = np.fft.ifft2(y_hat, axes=(-2, -1))
    return _unpack_2d_freq(y_grids, grid_rcv)


def _apply_2d_fft_kernel_cross_multi_freq(
    W_block_src: NDArray[np.complexfloating],
    kernel_hat_freq: NDArray[np.complexfloating],
    grid_src: NDArray[np.intp],
    grid_rcv: NDArray[np.intp],
    nP: int,
) -> NDArray[np.complexfloating]:
    """Frequency-batched multi-RHS 2D FFT convolution.

    Args:
        W_block_src: Source data, shape (F, M_src, 9, k).
        kernel_hat_freq: FFT of kernel, shape (F, 9, 9, nP, nP).
        grid_src: Source 2D grid indices, shape (M_src, 2).
        grid_rcv: Receiver 2D grid indices, shape (M_rcv, 2).
        nP: Padded 2D grid size.

    Returns:
        Result at receiver points, shape (F, M_rcv, 9, k).
    """
    grids = _pack_2d_multi_freq(W_block_src, grid_src, nP)  # (F, 9, k, nP, nP)
    w_hat = np.fft.fft2(grids, axes=(-2, -1))
    y_hat = np.einsum("fijxy,fjkxy->fikxy", kernel_hat_freq, w_hat)
    y_grids = np.fft.ifft2(y_hat, axes=(-2, -1))
    return _unpack_2d_multi_freq(y_grids, grid_rcv)


def _apply_2d_fft_kernel_cross_multi(
    W_block_src: NDArray[np.complexfloating],
    kernel_hat: NDArray[np.complexfloating],
    grid_src: NDArray[np.intp],
    grid_rcv: NDArray[np.intp],
    nP: int,
) -> NDArray[np.complexfloating]:
    """Multi-RHS version of :func:`_apply_2d_fft_kernel_cross`.

    Applies the 9×9 FFT convolution kernel to all ``k`` RHS columns in a
    single batched FFT, reusing Fix 1 across the RHS axis.

    Args:
        W_block_src: Source data, shape (M_src, 9, k).
        kernel_hat: FFT of kernel, shape (9, 9, nP, nP).
        grid_src: Source 2D grid indices, shape (M_src, 2).
        grid_rcv: Receiver 2D grid indices, shape (M_rcv, 2).
        nP: Padded 2D grid size.

    Returns:
        Result at receiver points, shape (M_rcv, 9, k).
    """
    grids = _pack_2d_multi(W_block_src, grid_src, nP)  # (9, k, nP, nP)
    w_hat = np.fft.fft2(grids, axes=(-2, -1))
    # Einsum contracts the 9×9 block along j and broadcasts over k RHS columns.
    y_hat = np.einsum("ijxy,jkxy->ikxy", kernel_hat, w_hat)
    y_grids = np.fft.ifft2(y_hat, axes=(-2, -1))
    return _unpack_2d_multi(y_grids, grid_rcv)


def layered_matvec_freq(
    W_flat: NDArray[np.complexfloating],
    decomp: LayerDecomposition,
    intralayer_kernels_freq: list[NDArray[np.complexfloating]],
    interlayer_kernels_freq: dict[int, NDArray[np.complexfloating]],
    n_sub: int,
) -> NDArray[np.complexfloating]:
    """Frequency-batched ``(I − G·T)·w`` matvec.

    Applies the layered Foldy-Lax operator to a ``(F, 9·N)`` state in
    one pass.  The layer decomposition is frequency-independent, so all
    per-layer stencils, slices and 2D grid indices are computed once and
    reused across ``F``.

    Args:
        W_flat: Input state, shape ``(F, 9·N_total)``, z-sorted order.
        decomp: Layer decomposition metadata.
        intralayer_kernels_freq: List of length ``n_layers``, each entry
            a ``(F, 9, 9, nP, nP)`` kernel tensor.
        interlayer_kernels_freq: Dict mapping signed Δz to
            ``(F, 9, 9, nP, nP)`` kernel tensors.
        n_sub: Grid size per edge.

    Returns:
        ``(F, 9·N_total)`` result in z-sorted order.
    """
    if W_flat.ndim != 2:
        msg = (
            f"W_flat must be 2D (F, 9*N); got ndim={W_flat.ndim}. "
            "Use layered_matvec_multi_freq for multi-RHS (F, 9*N, k)."
        )
        raise ValueError(msg)
    n_freq = W_flat.shape[0]
    nP = 2 * n_sub - 1
    result = W_flat.copy()

    # Reshape to (F, M_lz, 9) per layer once.
    layer_blocks: list[NDArray[np.complexfloating]] = []
    for lz in range(decomp.n_layers):
        s = decomp.layer_slices[lz]
        M = int(decomp.layer_sizes[lz])
        block = W_flat[:, 9 * s.start : 9 * s.stop].reshape(n_freq, M, 9)
        layer_blocks.append(block)

    for lr in range(decomp.n_layers):
        sr = decomp.layer_slices[lr]
        Mr = int(decomp.layer_sizes[lr])
        grid_r = decomp.layer_grid_2d[lr]
        accum = np.zeros((n_freq, Mr, 9), dtype=complex)

        for ls in range(decomp.n_layers):
            grid_s = decomp.layer_grid_2d[ls]
            dz = int(decomp.z_indices[lr] - decomp.z_indices[ls])
            if dz == 0:
                kernel_freq = intralayer_kernels_freq[ls]
            else:
                if dz not in interlayer_kernels_freq:
                    continue
                kernel_freq = interlayer_kernels_freq[dz]

            accum += _apply_2d_fft_kernel_cross_freq(
                layer_blocks[ls], kernel_freq, grid_s, grid_r, nP
            )

        result[:, 9 * sr.start : 9 * sr.stop] += accum.reshape(n_freq, 9 * Mr)

    return result


def layered_matvec_multi_freq(
    W_flat: NDArray[np.complexfloating],
    decomp: LayerDecomposition,
    intralayer_kernels_freq: list[NDArray[np.complexfloating]],
    interlayer_kernels_freq: dict[int, NDArray[np.complexfloating]],
    n_sub: int,
) -> NDArray[np.complexfloating]:
    """Frequency-batched multi-RHS ``(I − G·T)·W`` matvec.

    Applies the layered Foldy-Lax operator to a ``(F, 9·N, k)`` block
    in one pass.  Combines Fix 5 (block Krylov basis) with Fix 6
    (frequency axis): the same pack/FFT/einsum pipeline handles both the
    leading ``F`` and trailing ``k`` axes.

    Args:
        W_flat: Input state, shape ``(F, 9·N_total, k)``, z-sorted order.
        decomp: Layer decomposition metadata.
        intralayer_kernels_freq: List of length ``n_layers``.
        interlayer_kernels_freq: Dict mapping signed Δz to kernels.
        n_sub: Grid size per edge.

    Returns:
        ``(F, 9·N_total, k)`` result in z-sorted order.
    """
    if W_flat.ndim != 3:
        msg = f"W_flat must be 3D (F, 9*N, k); got ndim={W_flat.ndim}"
        raise ValueError(msg)
    n_freq, _, k_rhs = W_flat.shape
    nP = 2 * n_sub - 1
    result = W_flat.copy()

    layer_blocks: list[NDArray[np.complexfloating]] = []
    for lz in range(decomp.n_layers):
        s = decomp.layer_slices[lz]
        M = int(decomp.layer_sizes[lz])
        block = W_flat[:, 9 * s.start : 9 * s.stop, :].reshape(n_freq, M, 9, k_rhs)
        layer_blocks.append(block)

    for lr in range(decomp.n_layers):
        sr = decomp.layer_slices[lr]
        Mr = int(decomp.layer_sizes[lr])
        grid_r = decomp.layer_grid_2d[lr]
        accum = np.zeros((n_freq, Mr, 9, k_rhs), dtype=complex)

        for ls in range(decomp.n_layers):
            grid_s = decomp.layer_grid_2d[ls]
            dz = int(decomp.z_indices[lr] - decomp.z_indices[ls])
            if dz == 0:
                kernel_freq = intralayer_kernels_freq[ls]
            else:
                if dz not in interlayer_kernels_freq:
                    continue
                kernel_freq = interlayer_kernels_freq[dz]

            accum += _apply_2d_fft_kernel_cross_multi_freq(
                layer_blocks[ls], kernel_freq, grid_s, grid_r, nP
            )

        result[:, 9 * sr.start : 9 * sr.stop, :] += accum.reshape(n_freq, 9 * Mr, k_rhs)

    return result


def layered_matvec_multi(
    W_flat: NDArray[np.complexfloating],
    decomp: LayerDecomposition,
    intralayer_kernels: list[NDArray[np.complexfloating]],
    interlayer_kernels: dict[int, NDArray[np.complexfloating]],
    n_sub: int,
) -> NDArray[np.complexfloating]:
    """Multi-RHS version of :func:`layered_matvec`.

    Applies ``(I - G·T)`` to a matrix ``W`` with shape ``(9·N, k)`` in
    one pass, amortising pack/FFT/unpack over all ``k`` right-hand sides
    (Fix 5 — block GMRES).

    Args:
        W_flat: Input block, shape ``(9·N, k)``, in z-sorted order.
        decomp: Layer decomposition metadata.
        intralayer_kernels: Per-layer intra-layer FFT kernels.
        interlayer_kernels: Dict mapping signed Δz → inter-layer kernel.
        n_sub: Grid size per edge.

    Returns:
        ``(I - G·T) · W`` with shape ``(9·N, k)``, in z-sorted order.
    """
    if W_flat.ndim != 2:
        msg = f"W_flat must be 2D (9*N, k); got ndim={W_flat.ndim}"
        raise ValueError(msg)
    k_rhs = W_flat.shape[1]
    nP = 2 * n_sub - 1
    result = W_flat.copy()  # identity term

    # Extract per-layer (M_lz, 9, k) blocks
    layer_blocks: list[NDArray[np.complexfloating]] = []
    for lz in range(decomp.n_layers):
        s = decomp.layer_slices[lz]
        M = decomp.layer_sizes[lz]
        block = W_flat[9 * s.start : 9 * s.stop, :].reshape(M, 9, k_rhs)
        layer_blocks.append(block)

    for lr in range(decomp.n_layers):
        sr = decomp.layer_slices[lr]
        Mr = decomp.layer_sizes[lr]
        grid_r = decomp.layer_grid_2d[lr]
        accum = np.zeros((Mr, 9, k_rhs), dtype=complex)

        for ls in range(decomp.n_layers):
            grid_s = decomp.layer_grid_2d[ls]
            dz = int(decomp.z_indices[lr] - decomp.z_indices[ls])
            if dz == 0:
                kernel = intralayer_kernels[ls]
            else:
                if dz not in interlayer_kernels:
                    continue
                kernel = interlayer_kernels[dz]

            accum += _apply_2d_fft_kernel_cross_multi(
                layer_blocks[ls], kernel, grid_s, grid_r, nP
            )

        result[9 * sr.start : 9 * sr.stop, :] += accum.reshape(9 * Mr, k_rhs)

    return result


# ---------------------------------------------------------------------------
# 4c. Block GMRES for multi-RHS shared-operator solves — Fix 5
# ---------------------------------------------------------------------------


def block_gmres(
    matvec_multi,
    B: NDArray[np.complexfloating],
    x0: NDArray[np.complexfloating] | None = None,
    rtol: float = 1e-8,
    max_iter: int = 100,
) -> tuple[NDArray[np.complexfloating], int, float]:
    """Minimal block GMRES for multi-RHS with a shared operator.

    Solves ``A X = B`` where ``B`` is an ``(n, k)`` block right-hand side
    and ``A`` is applied via ``matvec_multi: (n, k) → (n, k)``.  Builds a
    block Krylov subspace shared across all ``k`` columns, so the cost
    amortises dramatically over separate single-RHS GMRES calls.

    This is a no-restart implementation — fine for problem sizes where a
    few tens of block iterations suffice.

    Args:
        matvec_multi: Callable ``(n, k) → (n, k)``.
        B: Right-hand side block, shape ``(n, k)``.
        x0: Initial guess, shape ``(n, k)`` (default zero).
        rtol: Relative residual tolerance on the block Frobenius norm.
        max_iter: Maximum block Arnoldi iterations.

    Returns:
        ``(X, iterations, final_rel_res)``.
    """
    n, k = B.shape
    X = np.zeros_like(B) if x0 is None else x0.copy()
    R0 = B - matvec_multi(X)
    beta0 = np.linalg.norm(R0)
    if beta0 == 0.0:
        return X, 0, 0.0

    # Initial block QR: V[0] S0 = R0
    V0, S0 = np.linalg.qr(R0)
    V_blocks: list[NDArray[np.complexfloating]] = [V0]
    H_blocks: dict[tuple[int, int], NDArray[np.complexfloating]] = {}

    rel_res = 1.0
    final_iter = 0
    Y = np.zeros((k, k), dtype=B.dtype)

    for j in range(max_iter):
        W = matvec_multi(V_blocks[j])
        # Block modified Gram-Schmidt
        for i in range(j + 1):
            Hij = V_blocks[i].conj().T @ W
            H_blocks[(i, j)] = Hij
            W = W - V_blocks[i] @ Hij
        # New orthonormal block
        Qnew, Rnew = np.linalg.qr(W)
        H_blocks[(j + 1, j)] = Rnew
        V_blocks.append(Qnew)

        # Assemble block Hessenberg matrix H_bar of shape ((j+2)k, (j+1)k)
        bm = j + 1
        H_bar = np.zeros(((bm + 1) * k, bm * k), dtype=B.dtype)
        for ii in range(bm + 1):
            for jj in range(bm):
                if (ii, jj) in H_blocks:
                    H_bar[ii * k : (ii + 1) * k, jj * k : (jj + 1) * k] = H_blocks[
                        (ii, jj)
                    ]
        rhs_blk = np.zeros(((bm + 1) * k, k), dtype=B.dtype)
        rhs_blk[:k, :] = S0

        Y, *_ = np.linalg.lstsq(H_bar, rhs_blk, rcond=None)
        res_blk = rhs_blk - H_bar @ Y
        res_norm = np.linalg.norm(res_blk)
        rel_res = float(res_norm / beta0)
        final_iter = bm
        if rel_res < rtol:
            break

    # X = X + [V_0 | V_1 | ... | V_{bm-1}] · Y
    V_stack = np.concatenate(V_blocks[:final_iter], axis=1)
    X = X + V_stack @ Y
    return X, final_iter, rel_res


# ---------------------------------------------------------------------------
# Fix 6 — Frequency-batched block GMRES
# ---------------------------------------------------------------------------


_BLOCK_GMRES_FREQ_DEFAULT_MAXITER = 30
_BLOCK_GMRES_FREQ_MEMORY_CAP_BYTES = 4 * 1024**3  # 4 GB


def block_gmres_freq(
    matvec_multi_freq,
    B: NDArray[np.complexfloating],
    x0: NDArray[np.complexfloating] | None = None,
    rtol: float = 1e-8,
    max_iter: int = _BLOCK_GMRES_FREQ_DEFAULT_MAXITER,
) -> tuple[NDArray[np.complexfloating], int, NDArray[np.floating]]:
    """Frequency-batched block GMRES with a shared block Krylov subspace.

    Solves ``A_f X_f = B_f`` for each frequency ``f`` simultaneously,
    where ``matvec_multi_freq`` applies the per-frequency operator to an
    ``(F, n, k)`` block in one call.  The block Krylov basis is built
    from the batched residual at each iteration, so one block Arnoldi
    step services every frequency; per-frequency convergence is tracked
    via the Frobenius residual norm.

    Unlike the scalar :func:`block_gmres`, the returned ``rel_res`` is a
    length-``F`` array.  Iteration terminates when the **maximum**
    per-frequency residual falls below ``rtol``.  There is no active-
    frequency deflation — converged frequencies keep participating in
    the shared matvec (TODO for a future fix).

    Memory-safety pre-flight: raises ``MemoryError`` if the Krylov basis
    stack would exceed 4 GB.  The dominant term is the stored
    ``V_blocks``: ``max_iter × F × n × k × 16 bytes``.

    Args:
        matvec_multi_freq: Callable ``(F, n, k) → (F, n, k)``.
        B: Right-hand side block, shape ``(F, n, k)``.
        x0: Initial guess, shape ``(F, n, k)`` (default zero).
        rtol: Relative residual tolerance on the per-ω Frobenius norm.
            Convergence requires ``max_f rel_res_f < rtol``.
        max_iter: Maximum block Arnoldi iterations.  Defaults to 30 to
            keep the Krylov basis footprint bounded.

    Returns:
        ``(X, iterations, rel_res_freq)`` where ``X`` has shape
        ``(F, n, k)``, ``iterations`` is the number of block Arnoldi
        steps taken, and ``rel_res_freq`` has shape ``(F,)``.

    Raises:
        ValueError: On malformed inputs.
        MemoryError: On pre-flight Krylov-basis size exceeding 4 GB.
    """
    if B.ndim != 3:
        msg = f"B must be 3D (F, n, k); got ndim={B.ndim}"
        raise ValueError(msg)
    n_freq, n_dim, k_rhs = B.shape
    if x0 is not None and x0.shape != B.shape:
        msg = f"x0 shape {x0.shape} must match B shape {B.shape}"
        raise ValueError(msg)

    # Memory pre-flight: V_blocks stores (max_iter + 1) basis blocks,
    # each (F, n, k) complex128.
    bytes_per_block = n_freq * n_dim * k_rhs * 16
    total_bytes = (max_iter + 1) * bytes_per_block
    if total_bytes > _BLOCK_GMRES_FREQ_MEMORY_CAP_BYTES:
        gb = total_bytes / 1024**3
        msg = (
            f"block_gmres_freq Krylov basis would use {gb:.2f} GB "
            f"(cap 4.00 GB) at max_iter={max_iter}, F={n_freq}, "
            f"n={n_dim}, k={k_rhs}. Reduce max_iter or batch fewer "
            "frequencies per call."
        )
        raise MemoryError(msg)

    X = np.zeros_like(B) if x0 is None else x0.copy()
    R0 = B - matvec_multi_freq(X)
    # Per-ω initial Frobenius norms for convergence tracking.
    beta0_freq = np.linalg.norm(R0, axis=(-2, -1))  # (F,)
    rel_res_freq = np.zeros(n_freq, dtype=float)
    if np.all(beta0_freq == 0.0):
        return X, 0, rel_res_freq

    # Avoid div-by-zero for any trivially-zero frequency.
    safe_beta0 = np.where(beta0_freq == 0.0, 1.0, beta0_freq)

    # Initial block QR per ω: V[0] S0 = R0, shapes (F, n, k) and (F, k, k).
    V0, S0 = np.linalg.qr(R0)
    V_blocks: list[NDArray[np.complexfloating]] = [V0]
    H_blocks: dict[tuple[int, int], NDArray[np.complexfloating]] = {}

    final_iter = 0
    Y = np.zeros((n_freq, k_rhs, k_rhs), dtype=B.dtype)

    for j in range(max_iter):
        W = matvec_multi_freq(V_blocks[j])  # (F, n, k)
        # Block modified Gram-Schmidt per ω with broadcasted matmul.
        for i in range(j + 1):
            # Hij = V[i]^H @ W  → (F, k, k)
            Hij = np.matmul(V_blocks[i].conj().transpose(0, 2, 1), W)
            H_blocks[(i, j)] = Hij
            W = W - np.matmul(V_blocks[i], Hij)
        # Block QR of the residual for a new orthonormal basis block.
        Qnew, Rnew = np.linalg.qr(W)  # broadcasts over F
        H_blocks[(j + 1, j)] = Rnew
        V_blocks.append(Qnew)

        bm = j + 1
        hb_rows = (bm + 1) * k_rhs
        hb_cols = bm * k_rhs
        H_bar = np.zeros((n_freq, hb_rows, hb_cols), dtype=B.dtype)
        for ii in range(bm + 1):
            for jj in range(bm):
                if (ii, jj) in H_blocks:
                    H_bar[
                        :,
                        ii * k_rhs : (ii + 1) * k_rhs,
                        jj * k_rhs : (jj + 1) * k_rhs,
                    ] = H_blocks[(ii, jj)]
        rhs_blk = np.zeros((n_freq, hb_rows, k_rhs), dtype=B.dtype)
        rhs_blk[:, :k_rhs, :] = S0

        # Per-ω least squares via broadcasted normal equations.  Robust
        # fallback to a per-ω lstsq loop if the normal-equations solve
        # introduces NaN (ill-conditioned H_bar).
        H_bar_H = H_bar.conj().transpose(0, 2, 1)
        normal_A = np.matmul(H_bar_H, H_bar)  # (F, hb_cols, hb_cols)
        normal_b = np.matmul(H_bar_H, rhs_blk)  # (F, hb_cols, k)
        try:
            Y = np.linalg.solve(normal_A, normal_b)
            if not np.all(np.isfinite(Y)):
                raise np.linalg.LinAlgError("non-finite Y")
        except np.linalg.LinAlgError:
            Y = np.zeros((n_freq, hb_cols, k_rhs), dtype=B.dtype)
            for f_idx in range(n_freq):
                Y_f, *_ = np.linalg.lstsq(H_bar[f_idx], rhs_blk[f_idx], rcond=None)
                Y[f_idx] = Y_f

        res_blk = rhs_blk - np.matmul(H_bar, Y)
        res_norm_freq = np.linalg.norm(res_blk, axis=(-2, -1))  # (F,)
        rel_res_freq = res_norm_freq / safe_beta0
        final_iter = bm
        if np.max(rel_res_freq) < rtol:
            break

    # Reconstruct solution: X += [V_0 | ... | V_{bm-1}] · Y, per ω.
    V_stack = np.concatenate(V_blocks[:final_iter], axis=-1)  # (F, n, bm*k)
    X = X + np.matmul(V_stack, Y)
    return X, final_iter, rel_res_freq


# ---------------------------------------------------------------------------
# Fix 6 — Frequency-batched incident field and top-level slab solver
# ---------------------------------------------------------------------------


def _build_incident_field_coupled_freq(
    centres: NDArray[np.floating],
    omegas: NDArray[np.complexfloating],
    ref: ReferenceMedium,
    k_hat: NDArray | None = None,
    wave_type: str = "S",
) -> NDArray[np.complexfloating]:
    """Frequency-batched coupled incident field ``ψ_inc``.

    Loops over ω calling :func:`_build_incident_field_coupled`; the
    per-frequency transcendentals make a tight vectorisation infeasible,
    but the batch API lets the block FL solver feed ``(F, 9·N, 9)`` RHS
    straight into :func:`block_gmres_freq`.

    Args:
        centres: Sub-cell centres, shape (N, 3).
        omegas: Complex angular frequencies, shape (F,).
        ref: Background medium.
        k_hat: Incident unit direction (default z-hat upstream).
        wave_type: 'S' or 'P'.

    Returns:
        Incident field stack, shape ``(F, 9·N, 9)``.
    """
    _validate_omegas(omegas)
    n_freq = omegas.shape[0]
    # Probe shape once via the first frequency to allocate the output.
    psi0 = _build_incident_field_coupled(
        centres, omegas[0], ref, k_hat=k_hat, wave_type=wave_type
    )
    out = np.empty((n_freq,) + psi0.shape, dtype=complex)
    out[0] = psi0
    for f_idx in range(1, n_freq):
        out[f_idx] = _build_incident_field_coupled(
            centres, omegas[f_idx], ref, k_hat=k_hat, wave_type=wave_type
        )
    return out


def solve_slab_foldy_lax_freq(
    M: int,
    N_z: int,
    a: float,
    omegas: NDArray[np.complexfloating],
    ref: ReferenceMedium,
    contrast: MaterialContrast,
    k_hat: NDArray | None = None,
    wave_type: str = "S",
    rtol: float = 1e-8,
    max_iter: int = _BLOCK_GMRES_FREQ_DEFAULT_MAXITER,
) -> tuple[
    NDArray[np.complexfloating],
    int,
    NDArray[np.floating],
]:
    """Frequency-batched block Foldy-Lax solve for an ``M×M×N_z`` slab.

    End-to-end Fix 6 driver: builds geometry once, ``T_loc_freq``,
    frequency-batched intra/inter-layer kernel stacks, and the coupled
    incident field stack, then hands everything to
    :func:`block_gmres_freq` for a single shared-Krylov solve across all
    frequencies.  The frequency axis amortises stencil construction,
    layer decomposition, FFT setup and Krylov basis overhead.

    Args:
        M: Lateral grid size.
        N_z: Number of z-layers.
        a: Cube half-width (m).
        omegas: Complex angular frequencies, shape (F,).
        ref: Background medium.
        contrast: Material contrast.
        k_hat: Incident unit direction (default z-hat).
        wave_type: 'S' or 'P'.
        rtol: Relative residual tolerance for block_gmres_freq.
        max_iter: Max block Arnoldi iterations.

    Returns:
        ``(T_comp_freq, iters, rel_res_freq)`` with
        ``T_comp_freq`` of shape ``(F, 9, 9)`` and ``rel_res_freq`` of
        shape ``(F,)``.
    """
    _validate_omegas(omegas)
    geom = cluster_from_slab(M, N_z, a)
    decomp = decompose_layers(geom)
    nC = len(geom.centres)
    dim = 9 * nC
    n_freq = omegas.shape[0]

    T_loc_freq = build_T_loc_freq(a, ref, contrast, omegas)

    # Frequency-batched kernel stacks; for a uniform slab in free space
    # all layers share the same intra-layer kernel tensor.
    shared_intralayer_freq = build_intralayer_fft_kernel_freq(
        M, a, T_loc_freq, omegas, ref
    )
    intralayer_freq = [shared_intralayer_freq] * decomp.n_layers
    max_dz = (
        int(decomp.z_indices[-1] - decomp.z_indices[0]) if decomp.n_layers > 1 else 0
    )
    interlayer_freq = build_interlayer_kernel_cache_freq(
        M, a, T_loc_freq, omegas, ref, max_dz
    )

    # Incident field in original ordering, then z-sort.
    psi_inc_freq = _build_incident_field_coupled_freq(
        geom.centres, omegas, ref, k_hat=k_hat, wave_type=wave_type
    )  # (F, 9*N, 9)
    psi_inc_sorted = np.empty_like(psi_inc_freq)
    for f_idx in range(n_freq):
        for col in range(9):
            psi_inc_sorted[f_idx, :, col] = _reorder_flat(
                psi_inc_freq[f_idx, :, col], decomp.sort_order, nC
            )

    def matvec_multi_freq(W: NDArray) -> NDArray:
        return layered_matvec_multi_freq(W, decomp, intralayer_freq, interlayer_freq, M)

    X_sorted, iters, rel_res_freq = block_gmres_freq(
        matvec_multi_freq,
        psi_inc_sorted,
        x0=psi_inc_sorted.copy(),
        rtol=rtol,
        max_iter=max_iter,
    )

    # Unsort and collapse to composite T-matrix per ω.
    T_comp_freq = np.zeros((n_freq, 9, 9), dtype=complex)
    psi_exc_freq = np.empty((n_freq, dim, 9), dtype=complex)
    for f_idx in range(n_freq):
        for col in range(9):
            psi_exc_freq[f_idx, :, col] = _reorder_flat(
                X_sorted[f_idx, :, col], decomp.unsort_order, nC
            )
        for n in range(nC):
            T_comp_freq[f_idx] += (
                T_loc_freq[f_idx] @ psi_exc_freq[f_idx, 9 * n : 9 * n + 9, :]
            )

    return T_comp_freq, iters, rel_res_freq


# ---------------------------------------------------------------------------
# Heterogeneous-contrast path: propagator-only FFT kernels + per-cube T
# ---------------------------------------------------------------------------
#
# The FFT acceleration only requires translation invariance of the Green's
# function ``G``.  The local T-operator is block-diagonal per cube and can
# be applied in real space before the convolution, decoupling the FFT
# kernel from spatially varying contrasts.  The code below mirrors the
# uniform-contrast path but returns propagator-only kernels (``+P_9x9(r)``
# without the ``-P @ T_loc`` contraction) and layers in a per-cube T-apply
# step in the matvec.


def build_T_loc_per_cube_freq(
    a: float,
    ref: ReferenceMedium,
    contrasts_per_cube: list[MaterialContrast],
    omegas: NDArray[np.complexfloating],
) -> NDArray[np.complexfloating]:
    """Frequency-batched per-cube local 9×9 T-matrix stack.

    Loops over cubes and frequencies, calling the existing Rayleigh
    sub-cell machinery for each ``(cube, ω)`` pair.  The returned tensor
    indexes as ``out[f, n]`` where ``n`` is the cube index in the same
    order as ``contrasts_per_cube``.

    Args:
        a: Cube half-width.
        ref: Background medium.
        contrasts_per_cube: Per-cube material contrasts, length ``nC``.
        omegas: Complex angular frequencies, shape (F,).

    Returns:
        Stack of per-cube local T-matrices, shape (F, nC, 9, 9), complex.
    """
    _validate_omegas(omegas)
    if len(contrasts_per_cube) == 0:
        msg = "contrasts_per_cube must contain at least one MaterialContrast"
        raise ValueError(msg)
    n_freq = omegas.shape[0]
    n_cubes = len(contrasts_per_cube)
    out = np.empty((n_freq, n_cubes, 9, 9), dtype=complex)
    for f_idx, om in enumerate(omegas):
        for n, contrast in enumerate(contrasts_per_cube):
            rayleigh = compute_cube_tmatrix(om, a, ref, contrast)
            out[f_idx, n] = _sub_cell_tmatrix_9x9(rayleigh, om, a)
    return out


def build_propagator_fft_kernel_freq(
    n_sub: int,
    a_sub: float,
    omegas: NDArray[np.complexfloating],
    ref: ReferenceMedium,
    dz_cubes: int = 0,
) -> NDArray[np.complexfloating]:
    """Frequency-batched propagator-only FFT kernel (Δz fixed).

    Builds ``FFT2(-P_9x9(r))`` on a circular ``(2·n_sub - 1)²`` embedding
    at the fixed ``Δz = dz_cubes`` grid separation, without any ``T_loc``
    contraction.  The minus sign is baked in so that the heterogeneous
    matvec has the same ``result = W + kernel ⊛ (T·W)`` sign convention
    as the uniform path.

    Args:
        n_sub: Grid size per edge.
        a_sub: Cube half-width (m).
        omegas: Complex angular frequencies, shape (F,).
        ref: Background medium.
        dz_cubes: Signed layer separation in grid units (0 for
            intra-layer, ±1, ±2, ... for inter-layer).

    Returns:
        Propagator FFT kernel, shape ``(F, 9, 9, nP, nP)``, complex.
    """
    _validate_omegas(omegas)
    dd = 2.0 * a_sub
    nP = 2 * n_sub - 1
    n_freq = omegas.shape[0]

    r_vecs, ix_flat, iy_flat = _stencil_offsets(n_sub, dz=dz_cubes)
    r_vecs[:, 0] *= dd
    r_vecs[:, 1] *= dd
    r_vecs[:, 2] *= dd

    kernel = np.zeros((n_freq, 9, 9, nP, nP), dtype=complex)
    for f_idx, om in enumerate(omegas):
        P_batch = _propagator_block_9x9_batch(r_vecs, om, ref)  # (N, 9, 9)
        # Bake the minus sign into the kernel so that the matvec sign
        # convention matches the uniform-contrast path (Δψ = -P·T·ψ).
        block_batch = -P_batch
        kernel[f_idx, :, :, ix_flat, iy_flat] = block_batch

    return np.fft.fft2(kernel, axes=(-2, -1))


def build_interlayer_propagator_kernel_cache_freq(
    n_sub: int,
    a_sub: float,
    omegas: NDArray[np.complexfloating],
    ref: ReferenceMedium,
    max_dz: int,
) -> dict[int, NDArray[np.complexfloating]]:
    """Frequency-batched cache of propagator-only inter-layer kernels.

    Returns ``FFT2(-P_9x9(r))`` for ``|Δz| = 1..max_dz`` with z-reflection
    symmetry exploited to halve the work.  The z-parity identity
    ``P(-Δz)[i, j] = η_i η_j · P(+Δz)[i, j]`` is a pure propagator
    property — unlike the uniform-contrast cache it requires **no**
    assumption on ``T_loc`` because ``T_loc`` has been removed entirely
    from this kernel builder.

    Args:
        n_sub: Grid size per edge.
        a_sub: Cube half-width (m).
        omegas: Complex angular frequencies, shape (F,).
        ref: Background medium.
        max_dz: Maximum ``|Δz|`` in grid units.

    Returns:
        Dict mapping signed ``Δz → (F, 9, 9, nP, nP)`` FFT kernel.
    """
    _validate_omegas(omegas)
    cache: dict[int, NDArray[np.complexfloating]] = {}
    sign_mask = _Z_PARITY_MASK_9x9[None, :, :, None, None]  # (1, 9, 9, 1, 1)

    for dz in range(1, max_dz + 1):
        kernel_pos = build_propagator_fft_kernel_freq(
            n_sub, a_sub, omegas, ref, dz_cubes=dz
        )
        cache[dz] = kernel_pos
        cache[-dz] = sign_mask * kernel_pos

    return cache


def _apply_T_per_cube_multi_het_freq(
    W_layer: NDArray[np.complexfloating],
    T_layer: NDArray[np.complexfloating],
) -> NDArray[np.complexfloating]:
    """Apply per-cube 9×9 T-matrices to a ``(F, M, 9, k)`` layer block.

    Block-diagonal matmul: ``out[f, n, :, :] = T_layer[f, n] @ W[f, n, :, :]``
    implemented as a single ``einsum``.

    Args:
        W_layer: Layer block, shape ``(F, M, 9, k)``.
        T_layer: Per-cube T stack for this layer, shape ``(F, M, 9, 9)``.

    Returns:
        T-applied block, shape ``(F, M, 9, k)``.
    """
    return np.einsum("fmij,fmjk->fmik", T_layer, W_layer)


def layered_matvec_multi_het_freq(
    W_flat: NDArray[np.complexfloating],
    decomp: LayerDecomposition,
    T_per_cube_freq_by_layer: list[NDArray[np.complexfloating]],
    intralayer_prop_kernels_freq: list[NDArray[np.complexfloating]],
    interlayer_prop_kernels_freq: dict[int, NDArray[np.complexfloating]],
    n_sub: int,
) -> NDArray[np.complexfloating]:
    """Heterogeneous-contrast frequency-batched multi-RHS matvec.

    Computes ``(I − G·diag(T_n))·W`` for a ``(F, 9·N_total, k)`` block
    with per-cube local T-matrices.  Applies T in real space first, then
    the propagator-only FFT kernels.

    Args:
        W_flat: Input state, shape ``(F, 9·N_total, k)``, z-sorted order.
        decomp: Layer decomposition metadata.
        T_per_cube_freq_by_layer: Per-layer per-cube T stack.
            ``T_per_cube_freq_by_layer[lz]`` has shape ``(F, M_lz, 9, 9)``
            and indexes cubes in the same z-sorted order as ``decomp``.
        intralayer_prop_kernels_freq: Per-layer propagator-only intra-layer
            FFT kernels, each ``(F, 9, 9, nP, nP)``.
        interlayer_prop_kernels_freq: Dict mapping signed Δz to
            propagator-only FFT kernels, each ``(F, 9, 9, nP, nP)``.
        n_sub: Grid size per edge.

    Returns:
        ``(F, 9·N_total, k)`` result in z-sorted order.
    """
    if W_flat.ndim != 3:
        msg = f"W_flat must be 3D (F, 9*N, k); got ndim={W_flat.ndim}"
        raise ValueError(msg)
    n_freq, _, k_rhs = W_flat.shape
    nP = 2 * n_sub - 1
    result = W_flat.copy()

    # First, apply per-cube T to each layer's block in real space and
    # cache the T-applied layer blocks (shape ``(F, M_lz, 9, k)``).
    T_layer_blocks: list[NDArray[np.complexfloating]] = []
    for lz in range(decomp.n_layers):
        s = decomp.layer_slices[lz]
        M = int(decomp.layer_sizes[lz])
        block = W_flat[:, 9 * s.start : 9 * s.stop, :].reshape(n_freq, M, 9, k_rhs)
        T_layer = T_per_cube_freq_by_layer[lz]  # (F, M, 9, 9)
        if T_layer.shape != (n_freq, M, 9, 9):
            msg = (
                f"T_per_cube_freq_by_layer[{lz}] shape {T_layer.shape} "
                f"does not match expected {(n_freq, M, 9, 9)}"
            )
            raise ValueError(msg)
        T_layer_blocks.append(_apply_T_per_cube_multi_het_freq(block, T_layer))

    # Then convolve the T-applied blocks by the propagator-only FFT
    # kernels, layer-by-layer.
    for lr in range(decomp.n_layers):
        sr = decomp.layer_slices[lr]
        Mr = int(decomp.layer_sizes[lr])
        grid_r = decomp.layer_grid_2d[lr]
        accum = np.zeros((n_freq, Mr, 9, k_rhs), dtype=complex)

        for ls in range(decomp.n_layers):
            grid_s = decomp.layer_grid_2d[ls]
            dz = int(decomp.z_indices[lr] - decomp.z_indices[ls])
            if dz == 0:
                kernel_freq = intralayer_prop_kernels_freq[ls]
            else:
                if dz not in interlayer_prop_kernels_freq:
                    continue
                kernel_freq = interlayer_prop_kernels_freq[dz]

            accum += _apply_2d_fft_kernel_cross_multi_freq(
                T_layer_blocks[ls], kernel_freq, grid_s, grid_r, nP
            )

        result[:, 9 * sr.start : 9 * sr.stop, :] += accum.reshape(n_freq, 9 * Mr, k_rhs)

    return result


def solve_slab_foldy_lax_freq_het(
    M: int,
    N_z: int,
    a: float,
    omegas: NDArray[np.complexfloating],
    ref: ReferenceMedium,
    contrasts_per_cube: list[MaterialContrast],
    k_hat: NDArray | None = None,
    wave_type: str = "S",
    rtol: float = 1e-8,
    max_iter: int = _BLOCK_GMRES_FREQ_DEFAULT_MAXITER,
) -> tuple[
    NDArray[np.complexfloating],
    int,
    NDArray[np.floating],
]:
    """Heterogeneous-contrast frequency-batched block Foldy-Lax solve.

    Drop-in heterogeneous twin of :func:`solve_slab_foldy_lax_freq`: the
    only change in the physics API is that ``contrast`` (one global
    ``MaterialContrast``) is replaced by ``contrasts_per_cube`` (a list
    of one ``MaterialContrast`` per cube).  The per-cube contrasts must
    be provided in the same canonical order as the cube centres returned
    by :func:`cluster_from_slab` — i.e. looping ``iz`` (outer), ``ix``,
    ``iy`` (innermost).

    The per-cube T-matrices are applied in real space inside the matvec
    before the propagator-only FFT convolution, so that the FFT kernel
    depends only on the translation-invariant background Green's
    function.  This decouples the FFT acceleration from T-uniformity.

    The composite T-matrix is accumulated per frequency via
    ``T_comp = Σ_n T_n @ ψ_exc[n]`` — the direct generalisation of the
    uniform-contrast accumulator ``Σ_n T_loc @ ψ_exc[n]``.

    Args:
        M: Lateral grid size.
        N_z: Number of z-layers.
        a: Cube half-width (m).
        omegas: Complex angular frequencies, shape (F,).
        ref: Background medium.
        contrasts_per_cube: Length-``M·M·N_z`` list of per-cube
            :class:`MaterialContrast` in :func:`cluster_from_slab` order.
        k_hat: Incident unit direction (default z-hat).
        wave_type: 'S' or 'P'.
        rtol: Relative residual tolerance for block_gmres_freq.
        max_iter: Max block Arnoldi iterations.

    Returns:
        ``(T_comp_freq, iters, rel_res_freq)`` with
        ``T_comp_freq`` of shape ``(F, 9, 9)`` and ``rel_res_freq`` of
        shape ``(F,)``.
    """
    _validate_omegas(omegas)
    n_expected = M * M * N_z
    if len(contrasts_per_cube) != n_expected:
        msg = (
            f"contrasts_per_cube has length {len(contrasts_per_cube)}, "
            f"expected M·M·N_z = {M}·{M}·{N_z} = {n_expected}. "
            "Provide one MaterialContrast per cube in cluster_from_slab "
            "canonical order (iz outer, ix middle, iy inner)."
        )
        raise ValueError(msg)

    geom = cluster_from_slab(M, N_z, a)
    decomp = decompose_layers(geom)
    nC = len(geom.centres)
    dim = 9 * nC
    n_freq = omegas.shape[0]

    # Per-cube T stack in original cluster_from_slab order,
    # shape (F, nC, 9, 9).
    T_per_cube_freq = build_T_loc_per_cube_freq(a, ref, contrasts_per_cube, omegas)

    # Re-sort per-cube T to z-sorted order and split per layer.
    T_sorted = T_per_cube_freq[:, decomp.sort_order, :, :]
    T_per_cube_by_layer: list[NDArray[np.complexfloating]] = []
    for lz in range(decomp.n_layers):
        s = decomp.layer_slices[lz]
        T_per_cube_by_layer.append(T_sorted[:, s, :, :])

    # Propagator-only FFT kernels: intra-layer shared across layers
    # (free-space propagator is translation-invariant), inter-layer cache
    # keyed by signed Δz.
    shared_intralayer_freq = build_propagator_fft_kernel_freq(
        M, a, omegas, ref, dz_cubes=0
    )
    intralayer_freq = [shared_intralayer_freq] * decomp.n_layers
    max_dz = (
        int(decomp.z_indices[-1] - decomp.z_indices[0]) if decomp.n_layers > 1 else 0
    )
    interlayer_freq = build_interlayer_propagator_kernel_cache_freq(
        M, a, omegas, ref, max_dz
    )

    # Incident field in original ordering, then z-sort.
    psi_inc_freq = _build_incident_field_coupled_freq(
        geom.centres, omegas, ref, k_hat=k_hat, wave_type=wave_type
    )  # (F, 9*N, 9)
    psi_inc_sorted = np.empty_like(psi_inc_freq)
    for f_idx in range(n_freq):
        for col in range(9):
            psi_inc_sorted[f_idx, :, col] = _reorder_flat(
                psi_inc_freq[f_idx, :, col], decomp.sort_order, nC
            )

    def matvec_multi_freq(W: NDArray) -> NDArray:
        return layered_matvec_multi_het_freq(
            W,
            decomp,
            T_per_cube_by_layer,
            intralayer_freq,
            interlayer_freq,
            M,
        )

    X_sorted, iters, rel_res_freq = block_gmres_freq(
        matvec_multi_freq,
        psi_inc_sorted,
        x0=psi_inc_sorted.copy(),
        rtol=rtol,
        max_iter=max_iter,
    )

    # Composite T-matrix: Σ_n T_n @ ψ_exc[n].  Both the T stack and the
    # solution are in z-sorted order, so accumulate directly on the
    # sorted representation.
    T_comp_freq = np.zeros((n_freq, 9, 9), dtype=complex)
    # X_sorted has shape (F, 9·nC, 9).  Reshape to (F, nC, 9, 9), then
    # contract with T_sorted (F, nC, 9, 9) by einsum over the cube axis.
    psi_exc_sorted_blocks = X_sorted.reshape(n_freq, nC, 9, 9)
    T_comp_freq = np.einsum("fnij,fnjk->fik", T_sorted, psi_exc_sorted_blocks)

    return T_comp_freq, iters, rel_res_freq


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
