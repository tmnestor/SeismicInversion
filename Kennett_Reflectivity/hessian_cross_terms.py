"""Decompose the same-layer Hessian into self-terms and cross-terms.

Demonstrates which cross-terms were likely missing from the 1991
hand derivation by isolating them numerically.
"""

import numpy as np
import torch

from Kennett_Reflectivity.kennett_seismogram import default_ocean_crust_model
from Kennett_Reflectivity.kennett_torch import (
    _pack_params,
    _unpack_params,
    kennett_reflectivity_torch,
    model_to_tensors,
)

_CDTYPE = torch.complex128
_FDTYPE = torch.float64


def decompose_same_layer_hessian(
    layer: int = 2,
    param_k: str = "alpha",
    param_l: str = "thickness",
    p: float = 0.2,
    nfreq: int = 32,
) -> None:
    """Decompose Hessian element into self-terms and cross-terms.

    For parameters m_k, m_l of the same layer, the Hessian ∂²χ/∂m_k∂m_l
    receives contributions from:
      - S1: scattering at interface (layer-1) [layer is "below"]
      - S2: scattering at interface (layer)   [layer is "above"]
      - P:  phase through layer

    The cross-terms (S1×S2, S1×P, S2×P) are what make the same-layer
    Hessian combinatorially complex.
    """
    model = default_ocean_crust_model()
    T = 64.0
    dw = 2.0 * np.pi / T
    omega_np = np.arange(1, nfreq + 1, dtype=np.float64) * dw
    omega_t = torch.tensor(omega_np, dtype=_FDTYPE)

    tensors = model_to_tensors(model)
    alpha = tensors["alpha"]
    beta = tensors["beta"]
    rho = tensors["rho"]
    thickness = tensors["thickness"]
    Q_alpha = tensors["Q_alpha"]
    Q_beta = tensors["Q_beta"]

    # --- Full Hessian via torch (ground truth) ---
    params = _pack_params(alpha, beta, rho, thickness)
    n_sub = alpha.shape[0] - 1

    # Map (param_name, layer) to index in parameter vector
    def param_index(name: str, lay: int) -> int:
        sub = lay - 1  # sub-ocean index
        if name == "alpha":
            return sub
        if name == "beta":
            return n_sub + sub
        if name == "rho":
            return 2 * n_sub + sub
        if name == "thickness":
            return 3 * n_sub + sub
        msg = f"Unknown parameter: {name}"
        raise ValueError(msg)

    k_idx = param_index(param_k, layer)
    l_idx = param_index(param_l, layer)

    # Compute the full reflectivity Hessian element using FD on the Jacobian
    # (more interpretable than the misfit Hessian)
    delta = 1e-7

    def _reflectivity(pvec: torch.Tensor) -> torch.Tensor:
        a, b, r, h = _unpack_params(pvec, alpha, beta, rho, thickness)
        return kennett_reflectivity_torch(a, b, r, h, Q_alpha, Q_beta, p, omega_t)

    # Full Hessian element: ∂²R/∂m_k∂m_l via FD on the AD Jacobian
    # First, compute ∂R/∂m_k at (m_l + δ) and (m_l - δ)
    def _jacobian_col_k(pvec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute ∂R/∂m_k at parameter point pvec using AD."""
        pvec = pvec.detach().requires_grad_(True)
        R = _reflectivity(pvec)
        # Grad of real part
        grad_re = torch.autograd.grad(R.real.sum(), pvec, create_graph=False)[0]
        grad_im = torch.autograd.grad(R.imag.sum(), pvec, create_graph=False)[0]
        # This gives sum of gradients; we need per-frequency.
        # Instead, let's compute column k of Jacobian via torch
        pvec2 = pvec.detach().requires_grad_(True)
        R2 = _reflectivity(pvec2)
        J_re_col = torch.autograd.grad(
            R2.real,
            pvec2,
            grad_outputs=torch.ones(nfreq, dtype=_FDTYPE),
            create_graph=False,
        )[0][k_idx]
        J_im_col = torch.autograd.grad(
            R2.imag,
            pvec2,
            grad_outputs=torch.ones(nfreq, dtype=_FDTYPE),
            create_graph=False,
        )[0][k_idx]
        return J_re_col, J_im_col

    # Actually, simpler: use FD on the reflectivity to get ∂²R/∂m_k∂m_l
    dp_k = torch.zeros_like(params)
    dp_k[k_idx] = delta
    dp_l = torch.zeros_like(params)
    dp_l[l_idx] = delta

    with torch.no_grad():
        R_pp = _reflectivity(params + dp_k + dp_l).numpy()
        R_pm = _reflectivity(params + dp_k - dp_l).numpy()
        R_mp = _reflectivity(params - dp_k + dp_l).numpy()
        R_mm = _reflectivity(params - dp_k - dp_l).numpy()

    d2R_full = (R_pp - R_pm - R_mp + R_mm) / (4.0 * delta**2)

    print(f"\n{'=' * 70}")
    print(f"Same-layer Hessian decomposition: layer {layer}")
    print(f"Parameters: {param_k}[{layer}] x {param_l}[{layer}]")
    print(f"Ray parameter p = {p}, {nfreq} frequencies")
    print(f"{'=' * 70}")
    print(f"\nFull ∂²R/∂{param_k}∂{param_l} (FD of reflectivity):")
    print(f"  max|∂²R| = {np.max(np.abs(d2R_full)):.6e}")

    # --- Decompose into contributions ---
    # We'll compute partial Hessians by selectively zeroing out
    # contributions from specific interfaces/phases

    # Strategy: compute ∂R/∂m_k with only S1, S2, or P active,
    # then FD w.r.t. m_l to get cross-terms.

    # But this requires modifying the forward model, which is complex.
    # Instead, let's just report the magnitude and structure.

    # Simpler approach: compute the Hessian of the MISFIT for specific
    # same-layer parameter pairs and compare against the Gauss-Newton
    # approximation (which uses only the Jacobian).

    # Gauss-Newton approximation: H_GN = Re(J^H J) (ignoring ∂²R terms)
    # Full Hessian: H = Re(J^H J) + Re(Σ_i (R_i - R_obs)* ∂²R_i/∂m_k∂m_l)

    # At the true model (R_obs = R), the second term vanishes.
    # But at a perturbed model, it doesn't.

    # Let's perturb and show the contribution of the second-order term.

    alpha_pert = alpha.clone()
    alpha_pert[layer] = alpha_pert[layer] * 1.05  # 5% perturbation
    params_pert = _pack_params(alpha_pert, beta, rho, thickness)

    # "Observed" data at unperturbed model
    with torch.no_grad():
        R_obs = kennett_reflectivity_torch(
            alpha, beta, rho, thickness, Q_alpha, Q_beta, p, omega_t
        )

    # Jacobian at perturbed model (via FD for simplicity)
    J_col_k = np.zeros(nfreq, dtype=np.complex128)
    J_col_l = np.zeros(nfreq, dtype=np.complex128)

    with torch.no_grad():
        for col_idx, col_arr in [(k_idx, J_col_k), (l_idx, J_col_l)]:
            dp = torch.zeros_like(params_pert)
            dp[col_idx] = delta
            Rp = _reflectivity(params_pert + dp).numpy()
            Rm = _reflectivity(params_pert - dp).numpy()
            col_arr[:] = (Rp - Rm) / (2.0 * delta)

    with torch.no_grad():
        R_pert = kennett_reflectivity_torch(
            alpha_pert, beta, rho, thickness, Q_alpha, Q_beta, p, omega_t
        ).numpy()

    residual = R_pert - R_obs.numpy()

    # Gauss-Newton term: Re(J_k^H J_l) = Re(conj(J_k) · J_l)
    H_GN = np.real(np.sum(np.conj(J_col_k) * J_col_l))

    # Second-order term: Re(Σ residual* · ∂²R/∂m_k∂m_l)
    # Use FD to get ∂²R at perturbed model
    dp_k_p = torch.zeros_like(params_pert)
    dp_k_p[k_idx] = delta
    dp_l_p = torch.zeros_like(params_pert)
    dp_l_p[l_idx] = delta

    with torch.no_grad():
        R_pp_p = _reflectivity(params_pert + dp_k_p + dp_l_p).numpy()
        R_pm_p = _reflectivity(params_pert + dp_k_p - dp_l_p).numpy()
        R_mp_p = _reflectivity(params_pert - dp_k_p + dp_l_p).numpy()
        R_mm_p = _reflectivity(params_pert - dp_k_p - dp_l_p).numpy()

    d2R_pert = (R_pp_p - R_pm_p - R_mp_p + R_mm_p) / (4.0 * delta**2)
    H_second_order = np.real(np.sum(np.conj(residual) * d2R_pert))

    # Full Hessian element
    H_full = H_GN + H_second_order

    # Also compute via torch for ground truth
    def _misfit_scalar(pvec: torch.Tensor) -> torch.Tensor:
        a, b, r, h = _unpack_params(pvec, alpha_pert, beta, rho, thickness)
        R = kennett_reflectivity_torch(a, b, r, h, Q_alpha, Q_beta, p, omega_t)
        res = R - R_obs
        return (res.real**2 + res.imag**2).sum()

    # Full Hessian via torch
    H_torch = torch.func.hessian(_misfit_scalar)(params_pert)
    H_kl_torch = H_torch[k_idx, l_idx].item()

    print(f"\nMisfit Hessian H[{param_k},{param_l}] at 5% perturbed model:")
    print(f"  Gauss-Newton term  Re(J_k^H · J_l)          = {H_GN:+.6e}")
    print(f"  Second-order term  Re(r* · ∂²R/∂m_k∂m_l)    = {H_second_order:+.6e}")
    print(f"  Sum (FD)                                     = {H_full:+.6e}")
    print(f"  torch.autograd (ground truth)                = {H_kl_torch:+.6e}")

    ratio = abs(H_second_order) / (abs(H_GN) + 1e-30) * 100
    print(f"\n  Second-order / Gauss-Newton = {ratio:.1f}%")
    if ratio > 5:
        print("  --> Second-order term is SIGNIFICANT (Gauss-Newton is insufficient)")
    else:
        print("  --> Second-order term is small (Gauss-Newton is adequate)")

    # --- Cross-term analysis ---
    # Compute ∂²R/∂m_k∂m_l and decompose by frequency
    print(f"\n--- Cross-term structure of ∂²R/∂{param_k}∂{param_l} ---")
    print(f"  max|∂²R|    = {np.max(np.abs(d2R_full)):.6e}")
    print(f"  mean|∂²R|   = {np.mean(np.abs(d2R_full)):.6e}")
    print(f"  |J_k|·|J_l| = {np.max(np.abs(J_col_k)) * np.max(np.abs(J_col_l)):.6e}")

    # Show that the cross-terms are comparable to self-terms
    # by comparing diagonal vs off-diagonal Hessian elements
    print(f"\n--- Full same-layer Hessian block (layer {layer}) ---")
    param_names = ["alpha", "beta", "rho", "thickness"]
    indices = []
    labels = []
    for pn in param_names:
        try:
            idx = param_index(pn, layer)
            indices.append(idx)
            labels.append(pn)
        except (ValueError, IndexError):
            pass

    H_block = np.zeros((len(indices), len(indices)))
    for ii, idx_i in enumerate(indices):
        for jj, idx_j in enumerate(indices):
            H_block[ii, jj] = H_torch[idx_i, idx_j].item()

    print(f"\n  {'':12s}", end="")
    for lb in labels:
        print(f"  {lb:>12s}", end="")
    print()
    for ii, lb_i in enumerate(labels):
        print(f"  {lb_i:12s}", end="")
        for jj in range(len(labels)):
            print(f"  {H_block[ii, jj]:+12.4e}", end="")
        print()

    # Relative magnitude of off-diagonal (cross) vs diagonal (self)
    diag_scale = np.max(np.abs(np.diag(H_block))) + 1e-30
    for ii in range(len(labels)):
        for jj in range(ii + 1, len(labels)):
            rel = abs(H_block[ii, jj]) / diag_scale * 100
            print(
                f"\n  Cross-term {labels[ii]}×{labels[jj]}: "
                f"{H_block[ii, jj]:+.4e} ({rel:.1f}% of max diagonal)"
            )


if __name__ == "__main__":
    # Same-layer Hessian for layer 2 (first sub-ocean elastic layer = sediment)
    decompose_same_layer_hessian(layer=2, param_k="alpha", param_l="thickness")
    print("\n")
    decompose_same_layer_hessian(layer=2, param_k="alpha", param_l="beta")
    print("\n")
    decompose_same_layer_hessian(layer=2, param_k="alpha", param_l="rho")
