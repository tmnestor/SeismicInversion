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
