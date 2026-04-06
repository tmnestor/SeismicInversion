"""Dressed T-matrix: folds intra-layer MS into an effective 9×9 T.

For a space-filling layer of cubes (d = 2a), the self-energy G_self
sums the 9×9 free-space propagators from a cube to all its horizontal
neighbours.  The dressed T-matrix

    T_dressed = T_bare · [I − G_self · T_bare]⁻¹

replaces T_bare, capturing all orders of intra-layer multiple
scattering.  Plugs directly into the Riccati interlayer solver.
"""

import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_MS_ROOT = Path("/Users/tod/Desktop/MultipleScatteringCalculations")
if str(_MS_ROOT) not in sys.path:
    sys.path.insert(0, str(_MS_ROOT))

from cubic_scattering import (  # noqa: E402
    MaterialContrast,
    ReferenceMedium,
    SlabGeometry,
    compute_slab_tmatrices,
    uniform_slab_material,
)
from cubic_scattering.resonance_tmatrix import _propagator_block_9x9  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from GlobalMatrix.block_riccati_cluster import (  # noqa: E402
    _propagator_block_9x9_batch,
    _validate_omegas,
)


def self_energy_greens_9x9(
    d: float,
    omega: complex,
    ref: ReferenceMedium,
    n_rings: int = 50,
) -> NDArray:
    """Self-energy: sum of 9×9 propagators over horizontal neighbours.

    Computes the k=0 lattice Green's function for a square lattice:

        G_self = Σ_{(m,n)≠(0,0)} G_0(0, m·d, n·d)

    summed over all integer offsets with max(|m|, |n|) ≤ n_rings.
    No Bloch phases — this is the self-energy for a scatterer
    embedded in a space-filling square lattice.

    Args:
        d: Lattice spacing (m).  For space-filling cubes, d = 2a.
        omega: Angular frequency (rad/s), may be complex.
        ref: Background elastic medium.
        n_rings: Truncation radius in lattice units.

    Returns:
        Self-energy, shape (9, 9), complex.
    """
    G_self = np.zeros((9, 9), dtype=complex)
    for m in range(-n_rings, n_rings + 1):
        for n in range(-n_rings, n_rings + 1):
            if m == 0 and n == 0:
                continue
            r_vec = np.array([0.0, m * d, n * d])
            G_self += _propagator_block_9x9(r_vec, omega, ref)  # type: ignore[arg-type]
    return G_self


def dressed_layer_tmatrix(
    a: float,
    ref: ReferenceMedium,
    contrast: MaterialContrast,
    omega: complex,
    n_rings: int = 50,
) -> NDArray:
    """Dressed T-matrix for a single cube type.

    Computes:

        T_dressed = T_bare · [I − G_self · T_bare]⁻¹

    where T_bare is the 9×9 single-cube T-matrix and G_self is the
    self-energy from ``self_energy_greens_9x9``.

    Multiply by n = 1/d² for the per-unit-area T used by the
    interlayer solver.

    Args:
        a: Cube half-width (m).  Cube side d = 2a.
        ref: Background elastic medium.
        contrast: Material contrast of the cube.
        omega: Angular frequency (rad/s), may be complex.
        n_rings: Neighbour rings for self-energy.

    Returns:
        Dressed T-matrix, shape (9, 9), complex.
    """
    d = 2.0 * a

    # Single-cube T-matrix
    geom = SlabGeometry(M=1, N_z=1, a=a)
    mat = uniform_slab_material(geom, ref, contrast)
    T_bare = compute_slab_tmatrices(geom, mat, omega)[0, 0, 0]  # type: ignore[arg-type]

    # Self-energy
    G_self = self_energy_greens_9x9(d, omega, ref, n_rings=n_rings)

    # Dressed T: T_bare @ [I − G_self @ T_bare]⁻¹
    I9 = np.eye(9, dtype=complex)
    T_dressed: NDArray = T_bare @ np.linalg.solve(I9 - G_self @ T_bare, I9)

    return T_dressed


# ---------------------------------------------------------------------------
# Fix 6 — Frequency-batched self-energy and dressed T
# ---------------------------------------------------------------------------


def self_energy_greens_9x9_freq(
    d: float,
    omegas: NDArray,
    ref: ReferenceMedium,
    n_rings: int = 50,
) -> NDArray:
    """Frequency-batched square-lattice self-energy.

    Computes

        G_self(ω) = Σ_{(m,n)≠(0,0)} G_0(0, m·d, n·d; ω)

    for each ω in one pass.  The lattice offsets ``(m, n)`` are
    precomputed once outside the ω loop, and
    :func:`_propagator_block_9x9_batch` is invoked once per ω to return
    a ``(N_offsets, 9, 9)`` stack which is then summed.

    Args:
        d: Lattice spacing (m).  For space-filling cubes, ``d = 2a``.
        omegas: Complex angular frequencies, shape (F,).
        ref: Background elastic medium.
        n_rings: Truncation radius in lattice units.

    Returns:
        Self-energy stack, shape ``(F, 9, 9)``, complex.
    """
    _validate_omegas(omegas)
    n_freq = omegas.shape[0]
    if n_rings < 0:
        msg = f"n_rings must be non-negative, got {n_rings}"
        raise ValueError(msg)

    # Precomputed lattice offsets (amortised across F).
    m_grid, n_grid = np.meshgrid(
        np.arange(-n_rings, n_rings + 1),
        np.arange(-n_rings, n_rings + 1),
        indexing="ij",
    )
    m_flat = m_grid.ravel()
    n_flat = n_grid.ravel()
    mask = (m_flat != 0) | (n_flat != 0)
    m_flat = m_flat[mask]
    n_flat = n_flat[mask]

    G_self_freq = np.zeros((n_freq, 9, 9), dtype=complex)
    if m_flat.size == 0:
        return G_self_freq

    r_vecs = np.zeros((m_flat.size, 3), dtype=float)
    r_vecs[:, 1] = m_flat * d
    r_vecs[:, 2] = n_flat * d

    for f_idx, om in enumerate(omegas):
        P_batch = _propagator_block_9x9_batch(r_vecs, om, ref)  # (N, 9, 9)
        G_self_freq[f_idx] = P_batch.sum(axis=0)

    return G_self_freq


def dressed_layer_tmatrix_freq(
    a: float,
    ref: ReferenceMedium,
    contrast: MaterialContrast,
    omegas: NDArray,
    n_rings: int = 50,
) -> NDArray:
    """Frequency-batched dressed layer T-matrix.

    Computes

        T_dressed(ω) = T_bare(ω) · [I − G_self(ω) · T_bare(ω)]⁻¹

    for each ω in ``omegas``.  ``T_bare`` is assembled via a Python loop
    over :func:`compute_slab_tmatrices`, since that routine embeds
    dispersion nonlinearly in ω.  ``G_self`` comes from
    :func:`self_energy_greens_9x9_freq`.  The matrix inversion step is a
    single broadcasted :func:`numpy.linalg.solve` over the leading
    ``F`` axis.

    Args:
        a: Cube half-width (m).  Cube side ``d = 2a``.
        ref: Background elastic medium.
        contrast: Material contrast of the cube.
        omegas: Complex angular frequencies, shape (F,).
        n_rings: Neighbour rings for self-energy.

    Returns:
        Dressed T-matrix stack, shape ``(F, 9, 9)``, complex.
    """
    _validate_omegas(omegas)
    n_freq = omegas.shape[0]
    d = 2.0 * a

    # Bare single-cube T-matrix per ω.
    geom = SlabGeometry(M=1, N_z=1, a=a)
    mat = uniform_slab_material(geom, ref, contrast)
    T_bare_freq = np.empty((n_freq, 9, 9), dtype=complex)
    for f_idx, om in enumerate(omegas):
        T_bare_freq[f_idx] = compute_slab_tmatrices(geom, mat, om)[0, 0, 0]  # type: ignore[arg-type]

    # Self-energy per ω (batched lattice offsets).
    G_self_freq = self_energy_greens_9x9_freq(d, omegas, ref, n_rings=n_rings)

    # (I - G·T) per ω, broadcasted linear solve.
    I9 = np.eye(9, dtype=complex)
    I9_freq = np.broadcast_to(I9, (n_freq, 9, 9))
    lhs = I9_freq - np.matmul(G_self_freq, T_bare_freq)  # (F, 9, 9)
    # Solve lhs @ X = I (per ω) to get the inverse, then left-multiply by T_bare.
    inv = np.linalg.solve(lhs, I9_freq)
    return np.matmul(T_bare_freq, inv)
