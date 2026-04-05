"""Tau-p domain inversion using the GMM Riccati-sweep forward model.

Uses the differentiable Global Matrix Method (Block-Riccati O(N) solver) as the
forward model and reuses the Newton-Levenberg-Marquardt machinery from the
Kennett package.  Implicit differentiation through ``torch.linalg.solve``
provides exact gradients and Hessians.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from Kennett_Reflectivity.kennett_seismogram import default_ocean_crust_model
from Kennett_Reflectivity.kennett_torch import (
    _pack_params,
    _unpack_params,
    model_to_tensors,
)
from Kennett_Reflectivity.taup_inversion import (
    InversionResult,
    _LATEX_RCPARAMS,
    _LAYER_NAMES,
    _tensors_to_layer_model,
    newton_lm_step,
    plot_convergence_curves,
    write_model_profiles_tikz,
    write_model_table_latex,
)

from .gmm_torch import gmm_reflectivity_torch

if TYPE_CHECKING:
    from Kennett_Reflectivity.inversion_config import InversionConfig
    from Kennett_Reflectivity.layer_model import LayerModel

__all__ = [
    "InversionResult",
    "_default_config",
    "compute_taup_traces",
    "invert_taup",
    "plot_convergence_curves",
    "plot_taup_traces",
    "taup_misfit_grad_hessian",
    "write_model_profiles_tikz",
    "write_model_table_latex",
]

logger = logging.getLogger(__name__)

_FDTYPE = torch.float64
_CDTYPE = torch.complex128


# ---------------------------------------------------------------------------
# Misfit, gradient, and Hessian
# ---------------------------------------------------------------------------


def _make_misfit_fn(
    alpha_ref: torch.Tensor,
    beta_ref: torch.Tensor,
    rho_ref: torch.Tensor,
    thickness_ref: torch.Tensor,
    Q_alpha: torch.Tensor,
    Q_beta: torch.Tensor,
    p_val: float,
    omega: torch.Tensor,
    R_obs_p: torch.Tensor,
):
    """Return a closure mapping a packed parameter vector to scalar L2 misfit."""

    def _misfit(params: torch.Tensor) -> torch.Tensor:
        a, b, r, h = _unpack_params(
            params,
            alpha_ref,
            beta_ref,
            rho_ref,
            thickness_ref,
        )
        R = gmm_reflectivity_torch(
            a,
            b,
            r,
            h,
            Q_alpha,
            Q_beta,
            p_val,
            omega,
        )
        residual = R - R_obs_p
        return (residual.real**2 + residual.imag**2).sum()

    return _misfit


def taup_misfit_grad_hessian(
    log_params: np.ndarray,
    alpha_ref: torch.Tensor,
    beta_ref: torch.Tensor,
    rho_ref: torch.Tensor,
    thickness_ref: torch.Tensor,
    Q_alpha: torch.Tensor,
    Q_beta: torch.Tensor,
    p_values: list[float],
    omega: torch.Tensor,
    R_obs: dict[float, torch.Tensor],
) -> tuple[float, np.ndarray, np.ndarray]:
    """Multi-slowness misfit, gradient, and Hessian in log-parameter space.

    Sums the L2 misfit over all slownesses, then transforms the gradient and
    Hessian from physical to log-parameter space via the chain rule through
    ``exp()``.

    Args:
        log_params: Log-space parameter vector, shape ``(n_params,)``.
        alpha_ref: Reference P-wave velocities (full model).
        beta_ref: Reference S-wave velocities (full model).
        rho_ref: Reference densities (full model).
        thickness_ref: Reference thicknesses (full model).
        Q_alpha: P-wave quality factors.
        Q_beta: S-wave quality factors.
        p_values: Slowness values (s/km) to sum over.
        omega: Angular frequency grid.
        R_obs: Observed reflectivity keyed by slowness.

    Returns:
        ``(chi, g_log, H_log)`` — misfit scalar, log-space gradient
        ``(n_params,)``, and log-space Hessian ``(n_params, n_params)``.
    """
    n_params = len(log_params)
    phys_params = np.exp(log_params)
    phys_tensor = torch.tensor(phys_params, dtype=_FDTYPE)

    total_chi = 0.0
    total_grad_phys = np.zeros(n_params)
    total_H_phys = np.zeros((n_params, n_params))

    for p_val in p_values:
        R_obs_p = R_obs[p_val]
        misfit_fn = _make_misfit_fn(
            alpha_ref,
            beta_ref,
            rho_ref,
            thickness_ref,
            Q_alpha,
            Q_beta,
            p_val,
            omega,
            R_obs_p,
        )

        # Gradient via backward pass
        params_g = phys_tensor.clone().detach().requires_grad_(True)
        chi_p = misfit_fn(params_g)
        (grad_p,) = torch.autograd.grad(chi_p, params_g)

        total_chi += chi_p.item()
        total_grad_phys += grad_p.detach().numpy()

        # Hessian via torch.func.hessian (vectorized jacrev(grad))
        H_p = torch.func.hessian(misfit_fn)(phys_tensor)
        total_H_phys += H_p.detach().numpy()

    # Log-space transformation:
    #   g_log_i = m_i * g_phys_i
    #   H_log_ij = m_i * H_phys_ij * m_j  +  delta_ij * g_phys_i * m_i
    m = phys_params
    g_log = m * total_grad_phys
    H_log = np.outer(m, m) * total_H_phys + np.diag(total_grad_phys * m)

    return total_chi, g_log, H_log


def _eval_misfit_only(
    log_params: np.ndarray,
    alpha_ref: torch.Tensor,
    beta_ref: torch.Tensor,
    rho_ref: torch.Tensor,
    thickness_ref: torch.Tensor,
    Q_alpha: torch.Tensor,
    Q_beta: torch.Tensor,
    p_values: list[float],
    omega: torch.Tensor,
    R_obs: dict[float, torch.Tensor],
) -> float:
    """Evaluate multi-slowness misfit without gradient/Hessian (line search)."""
    phys_params = np.exp(log_params)
    phys_tensor = torch.tensor(phys_params, dtype=_FDTYPE)

    total_chi = 0.0
    with torch.no_grad():
        for p_val in p_values:
            R_obs_p = R_obs[p_val]
            a, b, r, h = _unpack_params(
                phys_tensor,
                alpha_ref,
                beta_ref,
                rho_ref,
                thickness_ref,
            )
            R = gmm_reflectivity_torch(
                a,
                b,
                r,
                h,
                Q_alpha,
                Q_beta,
                p_val,
                omega,
            )
            residual = R - R_obs_p
            total_chi += (residual.real**2 + residual.imag**2).sum().item()

    return total_chi


# ---------------------------------------------------------------------------
# Inversion driver
# ---------------------------------------------------------------------------


def invert_taup(
    true_model: LayerModel,
    p_values: list[float],
    nfreq: int = 64,
    perturbation: float = 0.15,
    max_iter: int = 100,
    seed: int = 42,
    tol: float = 1e-8,
) -> InversionResult:
    """Run a Newton-LM inversion in the tau-p domain using GMM forward model.

    Generates synthetic observed data from *true_model* using the GMM
    Riccati-sweep solver, creates a perturbed initial model, and iterates
    a full-Newton solver with Levenberg-Marquardt damping in log-parameter
    space until convergence.

    Args:
        true_model: True earth model (generates synthetic observed data).
        p_values: Slowness grid (s/km).
        nfreq: Number of frequencies.
        perturbation: Fractional perturbation for the initial model.
        max_iter: Maximum Newton iterations.
        seed: Random seed for the initial perturbation.
        tol: Convergence tolerance on gradient norm and relative misfit.

    Returns:
        InversionResult with convergence history and recovered model.
    """
    # True model tensors
    true_t = model_to_tensors(true_model)
    alpha_ref = true_t["alpha"]
    beta_ref = true_t["beta"]
    rho_ref = true_t["rho"]
    thickness_ref = true_t["thickness"]
    Q_alpha = true_t["Q_alpha"]
    Q_beta = true_t["Q_beta"]

    # Frequency grid: omega_k = k * dw, k = 1..nfreq, T = 64 s
    T = 64.0
    dw = 2.0 * np.pi / T
    omega = torch.arange(1, nfreq + 1, dtype=_FDTYPE) * dw

    # Synthetic observed data at the true model (using GMM)
    R_obs: dict[float, torch.Tensor] = {}
    with torch.no_grad():
        for p_val in p_values:
            R_obs[p_val] = gmm_reflectivity_torch(
                alpha_ref,
                beta_ref,
                rho_ref,
                thickness_ref,
                Q_alpha,
                Q_beta,
                p_val,
                omega,
            ).clone()

    # True parameter vector (sub-ocean only)
    true_params = _pack_params(
        alpha_ref,
        beta_ref,
        rho_ref,
        thickness_ref,
    ).numpy()
    n_params = len(true_params)
    n_sub = true_model.n_layers - 1

    # Perturbed initial model
    rng = np.random.default_rng(seed)
    perturb = np.ones(n_params)
    n_vel = 2 * n_sub  # alpha_sub + beta_sub
    n_thick = n_sub - 1
    perturb[:n_vel] = rng.uniform(
        1.0 - perturbation,
        1.0 + perturbation,
        n_vel,
    )
    perturb[n_vel : n_vel + n_sub] = rng.uniform(
        1.0 - perturbation * 0.67,
        1.0 + perturbation * 0.67,
        n_sub,
    )
    perturb[n_vel + n_sub :] = rng.uniform(
        1.0 - perturbation * 1.33,
        1.0 + perturbation * 1.33,
        n_thick,
    )
    init_params = true_params * perturb

    # Build initial LayerModel for reporting
    init_pt = torch.tensor(init_params, dtype=_FDTYPE)
    a_i, b_i, r_i, h_i = _unpack_params(
        init_pt,
        alpha_ref,
        beta_ref,
        rho_ref,
        thickness_ref,
    )
    initial_model = _tensors_to_layer_model(
        a_i,
        b_i,
        r_i,
        h_i,
        true_model,
    )

    # Log-space starting point
    log_x = np.log(init_params)

    # Convergence history
    misfit_history: list[float] = []
    grad_norm_history: list[float] = []
    param_error_history: list[float] = []

    common_args = (
        alpha_ref,
        beta_ref,
        rho_ref,
        thickness_ref,
        Q_alpha,
        Q_beta,
        p_values,
        omega,
        R_obs,
    )

    lam: float | None = None

    for iteration in range(max_iter):
        chi, g, H = taup_misfit_grad_hessian(log_x, *common_args)

        grad_norm = float(np.linalg.norm(g))
        param_err = float(
            np.linalg.norm(np.exp(log_x) - true_params) / np.linalg.norm(true_params)
        )

        misfit_history.append(chi)
        grad_norm_history.append(grad_norm)
        param_error_history.append(param_err)

        logger.info(
            "iter %3d  chi=%.4e  ||g||=%.4e  err=%.4e",
            iteration,
            chi,
            grad_norm,
            param_err,
        )

        # Initialise damping from Hessian diagonal
        if lam is None:
            lam = 0.01 * max(float(np.abs(np.diag(H)).max()), 1e-10)

        # Convergence checks
        if grad_norm < tol:
            logger.info("Converged: gradient norm < tol")
            break
        if len(misfit_history) > 1:
            rel_change = abs(misfit_history[-2] - chi) / max(abs(chi), 1e-30)
            if rel_change < tol:
                logger.info("Converged: relative misfit change < tol")
                break

        # Newton-LM step with adaptive damping
        accepted = False
        for _ in range(10):
            dx = newton_lm_step(g, H, lam)
            trial_x = log_x + dx
            trial_chi = _eval_misfit_only(trial_x, *common_args)

            if trial_chi < chi:
                log_x = trial_x
                lam = max(lam / 2.0, 1e-15)
                accepted = True
                break
            lam *= 5.0

        if not accepted:
            logger.warning("Line search failed at iteration %d", iteration)
            break

    # Build recovered model
    recovered_params = np.exp(log_x)
    rec_pt = torch.tensor(recovered_params, dtype=_FDTYPE)
    a_r, b_r, r_r, h_r = _unpack_params(
        rec_pt,
        alpha_ref,
        beta_ref,
        rho_ref,
        thickness_ref,
    )
    recovered_model = _tensors_to_layer_model(
        a_r,
        b_r,
        r_r,
        h_r,
        true_model,
    )

    converged = bool(param_error_history and param_error_history[-1] < 0.01)

    return InversionResult(
        true_model=true_model,
        initial_model=initial_model,
        recovered_model=recovered_model,
        misfit_history=misfit_history,
        grad_norm_history=grad_norm_history,
        param_error_history=param_error_history,
        n_iterations=len(misfit_history),
        converged=converged,
    )


# ---------------------------------------------------------------------------
# Tau-p trace synthesis (using GMM NumPy forward model)
# ---------------------------------------------------------------------------


def compute_taup_traces(
    model: LayerModel,
    p_values: list[float],
    T: float = 64.0,
    nw: int = 1024,
) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    """Compute Ricker-convolved tau-p seismograms using the GMM forward model.

    Uses the NumPy GMM reflectivity (Riccati sweep) and the standard Ricker
    source followed by Hermitian-symmetric IFFT.

    Args:
        model: Stratified earth model.
        p_values: Slowness grid (s/km).
        T: Record length (seconds).
        nw: Number of positive frequencies (power of 2).

    Returns:
        ``(time, traces)`` where *time* has shape ``(2*nw,)`` and
        *traces* maps each slowness to a real seismogram array of the
        same length.
    """
    from Kennett_Reflectivity.source import ricker_spectrum

    from .global_matrix import gmm_reflectivity

    dw = 2.0 * np.pi / T
    nwm = nw - 1
    wmax = nw * dw
    omega = np.arange(1, nwm + 1, dtype=np.float64) * dw
    S = ricker_spectrum(omega, wmax)

    nt = 2 * nw
    dt = T / float(nt)
    time = np.arange(nt, dtype=np.float64) * dt

    traces: dict[float, np.ndarray] = {}
    for p_val in p_values:
        R = gmm_reflectivity(model, p_val, omega)
        Y = R * S
        Uwk = np.zeros(nt, dtype=np.complex128)
        Uwk[1:nw] = Y
        Uwk[nw + 1 :] = np.conj(Y[::-1])
        traces[p_val] = np.real(np.fft.fft(Uwk))

    return time, traces


# ---------------------------------------------------------------------------
# Tau-p trace plot
# ---------------------------------------------------------------------------


def plot_taup_traces(
    result: InversionResult,
    output: Path,
    p_values: list[float] | None = None,
    T: float = 64.0,
    nw: int = 1024,
    t_max: float = 15.0,
) -> None:
    """Save a tau-p trace comparison figure using GMM forward model.

    Left panel: variable-area wiggles with true (black) and recovered (red)
    overlaid.  Right panel: residual traces (blue), self-normalised to
    reveal structure.

    Args:
        result: Inversion result.
        output: Output path (``.pdf`` recommended).
        p_values: Slowness values for traces (default: 6 values 0.1--0.6).
        T: Record length (seconds).
        nw: Positive frequencies for seismogram synthesis.
        t_max: Maximum display time (seconds).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if p_values is None:
        p_values = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]

    _, traces_true = compute_taup_traces(result.true_model, p_values, T, nw)
    _, traces_rec = compute_taup_traces(
        result.recovered_model,
        p_values,
        T,
        nw,
    )
    time = np.arange(2 * nw, dtype=np.float64) * (T / (2 * nw))

    tmask = time <= t_max
    t_plot = time[tmask]

    global_peak = max(np.abs(traces_true[p][tmask]).max() for p in p_values)
    dp = p_values[1] - p_values[0] if len(p_values) > 1 else 0.1
    clip = 0.8

    with plt.rc_context(_LATEX_RCPARAMS):
        fig, (ax_main, ax_res) = plt.subplots(
            1,
            2,
            figsize=(14, 10),
            sharey=True,
        )

        # --- Left: overlaid traces ---
        for p_val in p_values:
            tr_t = traces_true[p_val][tmask].copy()
            tr_r = traces_rec[p_val][tmask].copy()
            if global_peak > 0:
                tr_t = tr_t / global_peak * dp * clip
                tr_r = tr_r / global_peak * dp * clip

            x0 = p_val
            ax_main.plot(x0 + tr_t, t_plot, "k-", linewidth=0.4)
            ax_main.fill_betweenx(
                t_plot,
                x0,
                x0 + tr_t,
                where=(tr_t > 0),
                interpolate=True,
                color="black",
                alpha=0.9,
            )
            ax_main.plot(
                x0 + tr_r,
                t_plot,
                color="#c0392b",
                linewidth=0.8,
            )

        ax_main.set_xlabel(r"Slowness $p$ (s/km)")
        ax_main.set_ylabel("Time (s)")
        ax_main.set_title("True (black) vs Recovered (red)")
        ax_main.invert_yaxis()
        ax_main.set_xlim(p_values[0] - dp, p_values[-1] + dp)

        # --- Right: residual traces ---
        res_peak = max(
            np.abs(
                traces_true[p][tmask] - traces_rec[p][tmask],
            ).max()
            for p in p_values
        )
        res_scale = global_peak if res_peak == 0 else res_peak

        for p_val in p_values:
            residual = traces_true[p_val][tmask] - traces_rec[p_val][tmask]
            if res_scale > 0:
                residual = residual / res_scale * dp * clip

            x0 = p_val
            ax_res.plot(
                x0 + residual,
                t_plot,
                color="#2980b9",
                linewidth=0.5,
            )
            ax_res.fill_betweenx(
                t_plot,
                x0,
                x0 + residual,
                where=(residual > 0),
                interpolate=True,
                color="#2980b9",
                alpha=0.6,
            )

        if global_peak > 0 and res_peak > 0:
            ratio = res_peak / global_peak
            ax_res.set_title(
                rf"Residual (peak $= {ratio:.1e} \times$ signal)",
            )
        else:
            ax_res.set_title("Residual")
        ax_res.set_xlabel(r"Slowness $p$ (s/km)")
        ax_res.set_xlim(p_values[0] - dp, p_values[-1] + dp)

        fig.tight_layout()
        plt.savefig(output, bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------


def _default_config() -> InversionConfig:
    """Return a default InversionConfig matching the original demo."""
    from Kennett_Reflectivity.inversion_config import (
        InversionConfig,
        OutputConfig,
        TraceDisplayConfig,
    )

    return InversionConfig(
        true_model=default_ocean_crust_model(),
        layer_names=list(_LAYER_NAMES),
        fixed_layers=[0],
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
        seed=42,
        tol=1e-8,
        output=OutputConfig(
            directory=Path("figures"),
            formats=["table", "profiles", "traces", "convergence"],
            trace_display=TraceDisplayConfig(t_max=15.0, nw=1024),
        ),
    )


def main() -> None:
    """Run a GMM tau-p inversion from a YAML config file or built-in defaults."""
    import argparse

    from Kennett_Reflectivity.inversion_config import load_config, save_config

    parser = argparse.ArgumentParser(
        description="GMM tau-p inversion with Newton-Levenberg-Marquardt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config (default: built-in 5-layer ocean-crust model)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help="Override max Newton iterations",
    )
    parser.add_argument(
        "--perturbation",
        type=float,
        default=None,
        help="Override initial model perturbation fraction",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Load config
    if args.config is not None:
        logger.info("Loading config: %s", args.config)
        cfg = load_config(args.config)
    else:
        logger.info("Using built-in default model")
        cfg = _default_config()

    # Apply CLI overrides
    if args.max_iter is not None:
        cfg.max_iter = args.max_iter
    if args.perturbation is not None:
        cfg.perturbation = args.perturbation
    if args.seed is not None:
        cfg.seed = args.seed
    if args.output_dir is not None:
        cfg.output.directory = args.output_dir

    # Run inversion
    logger.info("Running GMM tau-p inversion...")
    result = invert_taup(
        true_model=cfg.true_model,
        p_values=cfg.p_values,
        nfreq=cfg.nfreq,
        perturbation=cfg.perturbation,
        max_iter=cfg.max_iter,
        seed=cfg.seed,
        tol=cfg.tol,
    )

    # Print summary
    logger.info("Converged: %s", result.converged)
    logger.info("Iterations: %d", result.n_iterations)
    logger.info("Final misfit: %.2e", result.misfit_history[-1])
    logger.info("Final param error: %.4e", result.param_error_history[-1])

    # Save outputs
    outdir = cfg.output.directory
    outdir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    save_config(cfg, outdir / "gmm_inversion_config.yaml")
    logger.info("Config saved -> %s", outdir / "gmm_inversion_config.yaml")

    formats = cfg.output.formats

    if "table" in formats:
        write_model_table_latex(
            result,
            outdir / "gmm_taup_model_parameters.tex",
            layer_names=cfg.layer_names,
        )
        logger.info("LaTeX table  -> %s", outdir / "gmm_taup_model_parameters.tex")

    if "profiles" in formats:
        write_model_profiles_tikz(result, outdir / "gmm_taup_model_profiles.tex")
        logger.info("Depth profiles -> %s", outdir / "gmm_taup_model_profiles.tex")

    if "traces" in formats:
        td = cfg.output.trace_display
        plot_taup_traces(
            result,
            outdir / "gmm_taup_trace_comparison.pdf",
            T=64.0,
            nw=td.nw,
            t_max=td.t_max,
        )
        logger.info(
            "Trace comparison -> %s",
            outdir / "gmm_taup_trace_comparison.pdf",
        )

    if "convergence" in formats:
        plot_convergence_curves(result, outdir / "gmm_taup_convergence.pdf")
        logger.info("Convergence -> %s", outdir / "gmm_taup_convergence.pdf")


if __name__ == "__main__":
    main()
