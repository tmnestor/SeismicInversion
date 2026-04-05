"""
Kennett_Reflectivity: Kennett's recursive reflectivity method for stratified elastic media.

This package converts the Fortran program ``kennetslo.f`` to modern Python 3.12 with NumPy.
It computes plane wave reflectivity responses of stratified elastic half-spaces using
Kennett's Addition Formulae.

Main components:
  - LayerModel: Representation of stratified elastic media
  - Scattering matrices: P-SV interfacial reflection/transmission coefficients
  - Kennett reflectivity: Recursive algorithm for reflectivity computation
  - Seismogram generation: Synthetic seismogram computation and visualization
"""

from __future__ import annotations

from .kennett_reflectivity import batch_inv2x2, inv2x2, kennett_reflectivity
from .layer_model import LayerModel, complex_slowness, vertical_slowness
from .scattering_matrices import (
    ScatteringCoefficients,
    ocean_bottom_interface,
    solid_solid_interface,
)
from .source import ricker_spectrum, ricker_wavelet

__version__ = "1.0.0"
__author__ = "Converted from kennetslo.f (Kennett's method)"

__all__ = [
    "LayerModel",
    "complex_slowness",
    "vertical_slowness",
    "ScatteringCoefficients",
    "solid_solid_interface",
    "ocean_bottom_interface",
    "kennett_reflectivity",
    "inv2x2",
    "batch_inv2x2",
    "ricker_spectrum",
    "ricker_wavelet",
    "compute_seismogram",
    "compute_gather",
    "plot_gather",
    "default_ocean_crust_model",
    "compute_gather_gpu",
    "kennett_reflectivity_batch",
    "get_device",
    "kennett_reflectivity_torch",
    "forward_model_torch",
    "jacobian",
    "hessian",
    "model_to_tensors",
    "InversionResult",
    "compute_taup_traces",
    "invert_taup",
    "plot_convergence_curves",
    "write_model_profiles_tikz",
    "plot_taup_traces",
    "taup_misfit_grad_hessian",
    "write_model_table_latex",
    "InversionConfig",
    "OutputConfig",
    "TraceDisplayConfig",
    "load_config",
    "save_config",
]


def __getattr__(name: str):
    """Lazy-load submodules to avoid conflicts with ``python -m``."""
    _seismogram_names = ("compute_seismogram", "default_ocean_crust_model")
    _gather_names = ("compute_gather", "plot_gather")

    if name in _seismogram_names:
        from .kennett_seismogram import compute_seismogram, default_ocean_crust_model

        return {
            "compute_seismogram": compute_seismogram,
            "default_ocean_crust_model": default_ocean_crust_model,
        }[name]
    if name in _gather_names:
        from .kennett_gather import compute_gather, plot_gather

        return {"compute_gather": compute_gather, "plot_gather": plot_gather}[name]

    _gpu_names = ("compute_gather_gpu", "kennett_reflectivity_batch", "get_device")
    if name in _gpu_names:
        from .kennett_gather_gpu import compute_gather_gpu
        from .kennett_reflectivity_gpu import get_device, kennett_reflectivity_batch

        return {
            "compute_gather_gpu": compute_gather_gpu,
            "kennett_reflectivity_batch": kennett_reflectivity_batch,
            "get_device": get_device,
        }[name]

    _torch_names = (
        "kennett_reflectivity_torch",
        "forward_model_torch",
        "jacobian",
        "hessian",
        "model_to_tensors",
    )
    if name in _torch_names:
        from .kennett_torch import (
            forward_model_torch,
            hessian,
            jacobian,
            kennett_reflectivity_torch,
            model_to_tensors,
        )

        return {
            "kennett_reflectivity_torch": kennett_reflectivity_torch,
            "forward_model_torch": forward_model_torch,
            "jacobian": jacobian,
            "hessian": hessian,
            "model_to_tensors": model_to_tensors,
        }[name]

    _inversion_names = (
        "InversionResult",
        "compute_taup_traces",
        "invert_taup",
        "plot_convergence_curves",
        "write_model_profiles_tikz",
        "plot_taup_traces",
        "taup_misfit_grad_hessian",
        "write_model_table_latex",
    )
    if name in _inversion_names:
        from .taup_inversion import (
            InversionResult,
            compute_taup_traces,
            invert_taup,
            plot_convergence_curves,
            plot_taup_traces,
            taup_misfit_grad_hessian,
            write_model_profiles_tikz,
            write_model_table_latex,
        )

        return {
            "InversionResult": InversionResult,
            "compute_taup_traces": compute_taup_traces,
            "invert_taup": invert_taup,
            "plot_convergence_curves": plot_convergence_curves,
            "write_model_profiles_tikz": write_model_profiles_tikz,
            "plot_taup_traces": plot_taup_traces,
            "taup_misfit_grad_hessian": taup_misfit_grad_hessian,
            "write_model_table_latex": write_model_table_latex,
        }[name]

    _config_names = (
        "InversionConfig",
        "OutputConfig",
        "TraceDisplayConfig",
        "load_config",
        "save_config",
    )
    if name in _config_names:
        from .inversion_config import (
            InversionConfig,
            OutputConfig,
            TraceDisplayConfig,
            load_config,
            save_config,
        )

        return {
            "InversionConfig": InversionConfig,
            "OutputConfig": OutputConfig,
            "TraceDisplayConfig": TraceDisplayConfig,
            "load_config": load_config,
            "save_config": save_config,
        }[name]

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
