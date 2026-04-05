"""Tests for the SeismicInferenceNet architecture.

Verifies forward pass shapes, gradient flow, parameter count, and loss sanity.
"""

import math

import pytest
import torch

from NeuralInversion.inference_net import (
    ContinuousPositionalEncoding,
    SeismicInferenceNet,
    gaussian_nll_loss,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_P_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


@pytest.fixture()
def net():
    """Default SeismicInferenceNet with 12 slowness tokens, 64 frequencies."""
    return SeismicInferenceNet(
        n_p=12,
        nfreq=64,
        n_params=15,
        d_model=64,
        n_heads=4,
        n_layers=3,
        d_ff=128,
        dropout=0.0,
        p_values=_P_VALUES,
    )


@pytest.fixture()
def batch():
    """Random input batch: (batch=4, n_p=12, 2*nfreq=128)."""
    return torch.randn(4, 12, 128)


# ---------------------------------------------------------------------------
# Tests: Positional encoding
# ---------------------------------------------------------------------------


class TestContinuousPositionalEncoding:
    def test_output_shape(self):
        enc = ContinuousPositionalEncoding(d_model=64)
        p = torch.tensor(_P_VALUES)
        out = enc(p)
        assert out.shape == (12, 64)

    def test_different_p_different_encoding(self):
        enc = ContinuousPositionalEncoding(d_model=64)
        p = torch.tensor([0.1, 0.5])
        out = enc(p)
        assert not torch.allclose(out[0], out[1])

    def test_batched_p(self):
        enc = ContinuousPositionalEncoding(d_model=64)
        p = torch.tensor(_P_VALUES).unsqueeze(0).expand(3, -1)  # (3, 12)
        out = enc(p)
        assert out.shape == (3, 12, 64)


# ---------------------------------------------------------------------------
# Tests: Network forward pass
# ---------------------------------------------------------------------------


class TestForwardPass:
    def test_output_shapes(self, net, batch):
        mu, log_sigma = net(batch)
        assert mu.shape == (4, 15)
        assert log_sigma.shape == (4, 15)

    def test_single_sample(self, net):
        x = torch.randn(1, 12, 128)
        mu, log_sigma = net(x)
        assert mu.shape == (1, 15)
        assert log_sigma.shape == (1, 15)

    def test_log_sigma_clamped(self, net, batch):
        _, log_sigma = net(batch)
        assert log_sigma.min().item() >= -5.0
        assert log_sigma.max().item() <= 2.0

    def test_no_nan(self, net, batch):
        mu, log_sigma = net(batch)
        assert torch.all(torch.isfinite(mu))
        assert torch.all(torch.isfinite(log_sigma))

    def test_external_p_values(self, net, batch):
        """Forward pass with externally provided p_values."""
        p = torch.tensor(_P_VALUES)
        mu, log_sigma = net(batch, p_values=p)
        assert mu.shape == (4, 15)


# ---------------------------------------------------------------------------
# Tests: Gradient flow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    def test_all_params_grad(self, net, batch):
        mu, log_sigma = net(batch)
        loss = mu.sum() + log_sigma.sum()
        loss.backward()
        for name, param in net.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.all(torch.isfinite(param.grad)), (
                    f"Non-finite gradient for {name}"
                )

    def test_loss_backward(self, net, batch):
        z_true = torch.randn(4, 15)
        mu, log_sigma = net(batch)
        loss = gaussian_nll_loss(mu, log_sigma, z_true)
        loss.backward()
        # Check at least input_proj has gradient
        assert net.input_proj.weight.grad is not None
        assert torch.all(torch.isfinite(net.input_proj.weight.grad))


# ---------------------------------------------------------------------------
# Tests: Parameter count
# ---------------------------------------------------------------------------


class TestParameterCount:
    def test_total_params_reasonable(self, net):
        """Total params should be around 109k for the default architecture."""
        total = sum(p.numel() for p in net.parameters())
        assert 50_000 < total < 200_000, f"Unexpected param count: {total}"

    def test_has_expected_modules(self, net):
        assert hasattr(net, "input_proj")
        assert hasattr(net, "pos_enc")
        assert hasattr(net, "transformer")
        assert hasattr(net, "mu_head")
        assert hasattr(net, "log_sigma_head")


# ---------------------------------------------------------------------------
# Tests: Loss function
# ---------------------------------------------------------------------------


class TestGaussianNLLLoss:
    def test_zero_error_finite(self):
        mu = torch.zeros(10, 15)
        log_sigma = torch.zeros(10, 15)
        z_true = torch.zeros(10, 15)
        loss = gaussian_nll_loss(mu, log_sigma, z_true)
        assert torch.isfinite(loss)
        # With zero error and sigma=1, loss = sum(log(1)) / batch = 0
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_loss_with_error(self):
        mu = torch.zeros(10, 15)
        log_sigma = torch.zeros(10, 15)  # sigma = 1
        z_true = torch.ones(10, 15)
        loss = gaussian_nll_loss(mu, log_sigma, z_true)
        assert loss.item() > 0

    def test_smaller_sigma_higher_loss_at_error(self):
        """Smaller sigma should give higher loss when there is prediction error."""
        z_true = torch.ones(10, 15)
        mu = torch.zeros(10, 15)

        loss_wide = gaussian_nll_loss(mu, torch.ones(10, 15), z_true)  # sigma = e
        loss_narrow = gaussian_nll_loss(mu, -torch.ones(10, 15), z_true)  # sigma = 1/e
        assert loss_narrow.item() > loss_wide.item()

    def test_gradient_through_loss(self):
        mu = torch.randn(5, 15, requires_grad=True)
        log_sigma = torch.randn(5, 15, requires_grad=True)
        z_true = torch.randn(5, 15)
        loss = gaussian_nll_loss(mu, log_sigma, z_true)
        loss.backward()
        assert mu.grad is not None
        assert log_sigma.grad is not None

    def test_log_sigma_bias_init(self, net):
        """Verify log_sigma_head bias is initialised to log(0.1)."""
        expected = math.log(0.1)
        bias = net.log_sigma_head.bias.data
        assert torch.allclose(bias, torch.full_like(bias, expected))
