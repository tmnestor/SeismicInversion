"""Transformer-based inference network for seismic earth model parameters.

Architecture: SeismicInferenceNet
    Input R(omega, p) complex -> stack real/imag -> (batch, n_p, 2*nfreq)
    -> Linear projection -> + ContinuousPositionalEncoding(p_values)
    -> TransformerEncoder (pre-norm, 3 layers, 4 heads)
    -> mean pool over slowness tokens -> mu_head + log_sigma_head

Outputs log-space mean and standard deviation for 15 earth model parameters.
"""

import math

import torch
from torch import nn


class ContinuousPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from continuous physical slowness values.

    Unlike standard integer-position encodings, this uses the actual slowness
    value ``p`` to generate sinusoidal features, so the network generalises
    to unseen slowness grids.

    Args:
        d_model: Embedding dimension.
        max_freq_scale: Controls the range of sinusoidal frequencies.
    """

    def __init__(self, d_model: int, max_freq_scale: float = 10.0) -> None:
        super().__init__()
        self.d_model = d_model
        # Frequency bands: geometric spacing from 1 to max_freq_scale
        freqs = torch.exp(
            torch.linspace(
                math.log(1.0),
                math.log(max_freq_scale),
                d_model // 2,
            )
        )
        self.freqs: torch.Tensor
        self.register_buffer("freqs", freqs)  # (d_model // 2,)

    def forward(self, p_values: torch.Tensor) -> torch.Tensor:
        """Compute positional encoding for slowness values.

        Args:
            p_values: Slowness values, shape ``(n_p,)`` or ``(batch, n_p)``.

        Returns:
            Positional encoding, shape ``(..., n_p, d_model)``.
        """
        # p_values: (..., n_p) -> (..., n_p, 1)
        p = p_values.unsqueeze(-1)
        # freqs: (d_model // 2,)
        angles = p * self.freqs  # (..., n_p, d_model // 2)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class SeismicInferenceNet(nn.Module):
    """Transformer encoder mapping R(omega, p) to earth model parameter estimates.

    The network treats each slowness value as a token with a spectrum (real + imag
    components) as features.  A Transformer encoder learns cross-slowness (AVO)
    correlations.  Mean pooling over tokens followed by two linear heads produces
    log-space mean ``mu`` and log standard deviation ``log_sigma``.

    Args:
        n_p: Number of slowness values (tokens).
        nfreq: Number of frequency bins per slowness.
        n_params: Number of earth model parameters to predict.
        d_model: Transformer embedding dimension.
        n_heads: Number of attention heads.
        n_layers: Number of Transformer encoder layers.
        d_ff: Feed-forward dimension.
        dropout: Dropout rate.
        p_values: Slowness grid for positional encoding. If ``None``, no
            positional encoding is added.
    """

    def __init__(
        self,
        n_p: int = 12,
        nfreq: int = 64,
        n_params: int = 15,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 128,
        dropout: float = 0.1,
        p_values: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.n_p = n_p
        self.nfreq = nfreq
        self.n_params = n_params
        self.d_model = d_model

        # Input: real + imag stacked -> 2 * nfreq features per token
        self.input_proj = nn.Linear(2 * nfreq, d_model)

        # Positional encoding from physical slowness values
        self.pos_enc = ContinuousPositionalEncoding(d_model)
        if p_values is not None:
            self.register_buffer(
                "p_values",
                torch.tensor(p_values, dtype=torch.float32),
            )
        else:
            self.p_values: torch.Tensor | None = None

        # Pre-norm Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm for stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        # Output heads
        self.mu_head = nn.Linear(d_model, n_params)
        self.log_sigma_head = nn.Linear(d_model, n_params)

        # Initialise log_sigma bias to log(0.1) for stable early training
        nn.init.constant_(self.log_sigma_head.bias, math.log(0.1))

    def forward(
        self,
        x: torch.Tensor,
        p_values: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: reflectivity -> (mu, log_sigma).

        Args:
            x: Input reflectivity, shape ``(batch, n_p, 2*nfreq)`` float32.
                Real and imaginary parts stacked along the last dimension.
            p_values: Optional slowness values for positional encoding,
                shape ``(n_p,)``.  Overrides stored values if provided.

        Returns:
            ``(mu, log_sigma)`` each of shape ``(batch, n_params)``.
            ``mu`` is the predicted log-space parameter mean;
            ``log_sigma`` is the predicted log standard deviation (clamped to [-5, 2]).
        """
        # Input projection: (batch, n_p, 2*nfreq) -> (batch, n_p, d_model)
        h = self.input_proj(x)

        # Add positional encoding from slowness values
        p = p_values if p_values is not None else self.p_values
        if p is not None:
            h = h + self.pos_enc(p)

        # Transformer encoder
        h = self.transformer(h)  # (batch, n_p, d_model)

        # Mean pool over tokens
        h = h.mean(dim=1)  # (batch, d_model)

        mu = self.mu_head(h)
        log_sigma = self.log_sigma_head(h)
        log_sigma = log_sigma.clamp(-5.0, 2.0)

        return mu, log_sigma


def gaussian_nll_loss(
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    z_true: torch.Tensor,
) -> torch.Tensor:
    """Diagonal Gaussian negative log-likelihood loss.

    Equivalent to the ELBO under simulation-based amortized variational inference
    with a diagonal Gaussian approximate posterior.

    .. math::

        L = \\sum_i \\left[ \\frac{(z_{\\mathrm{true},i} - \\mu_i)^2}{2\\sigma_i^2}
            + \\log \\sigma_i \\right]

    Args:
        mu: Predicted means, shape ``(batch, n_params)``.
        log_sigma: Predicted log standard deviations, shape ``(batch, n_params)``.
        z_true: True log-space parameters, shape ``(batch, n_params)``.

    Returns:
        Scalar mean loss over the batch.
    """
    sigma = torch.exp(log_sigma)
    nll = 0.5 * ((z_true - mu) / sigma) ** 2 + log_sigma
    return nll.sum(dim=-1).mean()
