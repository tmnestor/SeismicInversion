"""Convergence and trace-recovery tests for the tau-p Newton-LM inversion."""

from __future__ import annotations

import numpy as np
import pytest

from Kennett_Reflectivity.kennett_seismogram import default_ocean_crust_model
from Kennett_Reflectivity.taup_inversion import compute_taup_traces, invert_taup


class TestTaupInversion:
    """Verify that the Newton-LM inversion recovers a known 3-layer sub-ocean model."""

    @pytest.mark.slow
    def test_convergence_15pct_perturbation(self):
        """Newton inversion converges from 15% perturbation to < 1% error."""
        result = invert_taup(
            true_model=default_ocean_crust_model(),
            p_values=[
                0.05,
                0.10,
                0.15,
                0.20,
                0.25,
                0.30,
                0.35,
                0.40,
                0.45,
                0.50,
                0.55,
                0.60,
            ],
            nfreq=64,
            perturbation=0.15,
            max_iter=50,
        )
        assert result.converged
        assert result.param_error_history[-1] < 0.01

    def test_convergence_small_perturbation(self):
        """Newton inversion converges from 5% perturbation to < 0.1% error."""
        result = invert_taup(
            true_model=default_ocean_crust_model(),
            p_values=[0.1, 0.2, 0.3, 0.4],
            nfreq=32,
            perturbation=0.05,
            max_iter=30,
        )
        assert result.converged
        assert result.param_error_history[-1] < 0.001

    def test_quadratic_convergence(self):
        """From 2% perturbation, converges to near-machine-precision recovery."""
        result = invert_taup(
            true_model=default_ocean_crust_model(),
            p_values=[0.1, 0.2, 0.3, 0.4],
            nfreq=32,
            perturbation=0.02,
            max_iter=20,
        )
        assert result.converged
        assert result.n_iterations <= 20
        # Quadratic convergence drives param error to near machine precision
        assert result.param_error_history[-1] < 1e-6


class TestTraceRecovery:
    """Compare Ricker-convolved tau-p seismograms: true vs recovered model."""

    @pytest.fixture
    def converged_result(self):
        """Run a 5% perturbation inversion (shared across trace tests)."""
        return invert_taup(
            true_model=default_ocean_crust_model(),
            p_values=[0.1, 0.2, 0.3, 0.4],
            nfreq=32,
            perturbation=0.05,
            max_iter=30,
        )

    def test_subcritical_traces(self, converged_result):
        """Recovered traces match at subcritical slownesses (p < 1/alpha_max)."""
        result = converged_result
        assert result.converged

        # p=0.10, 0.15 are subcritical for all sub-ocean layers
        p_test = [0.10, 0.15]
        _, traces_true = compute_taup_traces(
            result.true_model,
            p_test,
            nw=512,
        )
        _, traces_rec = compute_taup_traces(
            result.recovered_model,
            p_test,
            nw=512,
        )

        for p in p_test:
            signal = np.linalg.norm(traces_true[p])
            err = np.linalg.norm(traces_rec[p] - traces_true[p])
            assert err / signal < 0.01, (
                f"p={p}: relative trace error {err / signal:.2e}"
            )

    def test_postcritical_traces(self, converged_result):
        """Recovered traces match at post-critical slownesses."""
        result = converged_result
        assert result.converged

        # p=0.35, 0.45 are post-critical for crust (1/3.0=0.33) and
        # upper mantle (1/5.0=0.20)
        p_test = [0.35, 0.45]
        _, traces_true = compute_taup_traces(
            result.true_model,
            p_test,
            nw=512,
        )
        _, traces_rec = compute_taup_traces(
            result.recovered_model,
            p_test,
            nw=512,
        )

        for p in p_test:
            signal = np.linalg.norm(traces_true[p])
            err = np.linalg.norm(traces_rec[p] - traces_true[p])
            assert err / signal < 0.01, (
                f"p={p}: relative trace error {err / signal:.2e}"
            )

    def test_traces_at_inversion_slownesses(self, converged_result):
        """Traces match at the same slownesses used during inversion."""
        result = converged_result
        assert result.converged

        p_test = [0.1, 0.2, 0.3, 0.4]
        _, traces_true = compute_taup_traces(
            result.true_model,
            p_test,
            nw=512,
        )
        _, traces_rec = compute_taup_traces(
            result.recovered_model,
            p_test,
            nw=512,
        )

        for p in p_test:
            signal = np.linalg.norm(traces_true[p])
            err = np.linalg.norm(traces_rec[p] - traces_true[p])
            assert err / signal < 0.001, (
                f"p={p}: relative trace error {err / signal:.2e}"
            )
