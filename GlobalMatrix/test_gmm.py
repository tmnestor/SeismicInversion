"""Validate GMM forward model against Kennett reflectivity."""

import numpy as np
import pytest

from Kennett_Reflectivity.kennett_reflectivity import kennett_reflectivity
from Kennett_Reflectivity.kennett_seismogram import default_ocean_crust_model

from .global_matrix import gmm_reflectivity


@pytest.fixture
def model():
    return default_ocean_crust_model()


@pytest.fixture
def omega():
    T = 64.0
    nw = 256
    dw = 2.0 * np.pi / T
    return np.arange(1, nw, dtype=np.float64) * dw


class TestGMMvsKennett:
    """Verify that GMM and Kennett produce identical reflectivity."""

    @pytest.mark.parametrize("p", [0.1, 0.2, 0.3, 0.4, 0.6])
    def test_reflectivity_matches(self, model, omega, p):
        """GMM R(w,p) matches Kennett to < 1e-12 relative error."""
        R_kennett = kennett_reflectivity(model, p, omega, free_surface=False)
        R_gmm = gmm_reflectivity(model, p, omega, free_surface=False)

        # Relative error where |R| is not tiny
        mask = np.abs(R_kennett) > 1e-15
        rel_err = np.abs(R_gmm[mask] - R_kennett[mask]) / np.abs(R_kennett[mask])
        assert np.max(rel_err) < 1e-10, f"Max relative error: {np.max(rel_err):.2e}"

        # Absolute error everywhere
        abs_err = np.abs(R_gmm - R_kennett)
        assert np.max(abs_err) < 1e-12, f"Max absolute error: {np.max(abs_err):.2e}"

    @pytest.mark.parametrize("p", [0.1, 0.3, 0.6])
    def test_free_surface_matches(self, model, omega, p):
        """Free-surface mode matches Kennett."""
        R_kennett = kennett_reflectivity(model, p, omega, free_surface=True)
        R_gmm = gmm_reflectivity(model, p, omega, free_surface=True)

        mask = np.abs(R_kennett) > 1e-15
        rel_err = np.abs(R_gmm[mask] - R_kennett[mask]) / np.abs(R_kennett[mask])
        assert np.max(rel_err) < 1e-10, f"Max relative error: {np.max(rel_err):.2e}"

    @pytest.mark.parametrize("p", [0.1, 0.4, 0.8])
    def test_numerical_stability(self, model, omega, p):
        """All output values are finite (no overflow from evanescent modes)."""
        R = gmm_reflectivity(model, p, omega, free_surface=False)
        assert np.all(np.isfinite(R)), "Non-finite values in reflectivity"

    def test_dc_excluded(self, model):
        """Omega=0 would be singular; verify normal freqs work."""
        omega = np.array([0.1, 1.0, 10.0])
        R = gmm_reflectivity(model, 0.2, omega)
        assert np.all(np.isfinite(R))
