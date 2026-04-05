"""Global Matrix Method for plane-wave reflectivity in stratified elastic media.

Assembles all interface conditions into a single block-tridiagonal linear system
G x = b and solves for wave amplitudes. Mathematically equivalent to Kennett's
recursive method, but with cheaper derivatives: factor G once and back-substitute
for each parameter perturbation.

Convention: exp(-iwt) inverse Fourier transform, depth positive downward.
(Chin, Hedstrom & Thigpen (1984) use the conjugate exp(+iwt); all formulas here
are adapted to the Kennett convention.)

Reference: Chin, Hedstrom & Thigpen (1984), "Matrix methods in synthetic
seismograms", Geophys. J. R. astr. Soc.
"""

from .global_matrix import gmm_reflectivity
from .layer_matrix import (
    layer_eigenvectors,
    layer_eigenvectors_batched,
    layer_eigenvectors_sh_batched,
    layer_eigenvectors_torch,
    ocean_eigenvectors,
    ocean_eigenvectors_batched,
    ocean_eigenvectors_torch,
)
from .layered_greens import layered_greens_6x6, layered_greens_9x9, layered_greens_psv
from .riccati_solver import compute_source_vector

__version__ = "1.0.0"

__all__ = [
    "compute_source_vector",
    "gmm_reflectivity",
    "gmm_reflectivity_torch",
    "gmm_jacobian",
    "gmm_hessian",
    "layer_eigenvectors",
    "layer_eigenvectors_batched",
    "layer_eigenvectors_sh_batched",
    "layer_eigenvectors_torch",
    "layered_greens_6x6",
    "layered_greens_9x9",
    "layered_greens_psv",
    "ocean_eigenvectors",
    "ocean_eigenvectors_batched",
    "ocean_eigenvectors_torch",
    "compute_and_compare",
    "GMMConfig",
    "OutputConfig",
    "load_config",
    "save_config",
    "invert_taup",
    "InversionResult",
    "compute_taup_traces",
    "InterlayerMSResult",
    "InterlayerMSResult9x9",
    "ScattererSlab",
    "ScattererSlab9x9",
    "interlayer_ms_reflectivity",
    "interlayer_ms_reflectivity_9x9",
]


def __getattr__(name: str):
    """Lazy-load submodules to avoid import cost when not needed."""
    _torch_names = ("gmm_reflectivity_torch", "gmm_jacobian", "gmm_hessian")
    if name in _torch_names:
        from .gmm_torch import gmm_hessian, gmm_jacobian, gmm_reflectivity_torch

        return {
            "gmm_reflectivity_torch": gmm_reflectivity_torch,
            "gmm_jacobian": gmm_jacobian,
            "gmm_hessian": gmm_hessian,
        }[name]

    _config_names = ("GMMConfig", "OutputConfig", "load_config", "save_config")
    if name in _config_names:
        from .config import GMMConfig, OutputConfig, load_config, save_config

        return {
            "GMMConfig": GMMConfig,
            "OutputConfig": OutputConfig,
            "load_config": load_config,
            "save_config": save_config,
        }[name]

    _inversion_names = ("invert_taup", "InversionResult", "compute_taup_traces")
    if name in _inversion_names:
        from .taup_inversion import InversionResult, compute_taup_traces, invert_taup

        return {
            "invert_taup": invert_taup,
            "InversionResult": InversionResult,
            "compute_taup_traces": compute_taup_traces,
        }[name]

    _cli_names = ("compute_and_compare",)
    if name in _cli_names:
        from .gmm_reflectivity_cli import compute_and_compare

        return {"compute_and_compare": compute_and_compare}[name]

    _ms_names = (
        "InterlayerMSResult",
        "InterlayerMSResult9x9",
        "ScattererSlab",
        "ScattererSlab9x9",
        "interlayer_ms_reflectivity",
        "interlayer_ms_reflectivity_9x9",
    )
    if name in _ms_names:
        from .interlayer_ms import (
            InterlayerMSResult,
            InterlayerMSResult9x9,
            ScattererSlab,
            ScattererSlab9x9,
            interlayer_ms_reflectivity,
            interlayer_ms_reflectivity_9x9,
        )

        return {
            "InterlayerMSResult": InterlayerMSResult,
            "InterlayerMSResult9x9": InterlayerMSResult9x9,
            "ScattererSlab": ScattererSlab,
            "ScattererSlab9x9": ScattererSlab9x9,
            "interlayer_ms_reflectivity": interlayer_ms_reflectivity,
            "interlayer_ms_reflectivity_9x9": interlayer_ms_reflectivity_9x9,
        }[name]

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg) from None
