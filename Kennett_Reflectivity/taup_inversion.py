"""Tau-p domain inversion with Newton-Levenberg-Marquardt and exact Hessian.

Uses the differentiable Kennett reflectivity forward model to recover a layered
earth model from frequency-domain plane-wave reflectivity data R(omega, p).

The inversion operates in log-parameter space to enforce positivity and improve
Hessian conditioning.  The full Hessian is computed via ``torch.autograd`` and
used in a Levenberg-Marquardt trust-region scheme.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from .kennett_seismogram import default_ocean_crust_model
from .kennett_torch import (
    _pack_params,
    _unpack_params,
    kennett_reflectivity_torch,
    model_to_tensors,
)
from .layer_model import LayerModel

if TYPE_CHECKING:
    from .inversion_config import InversionConfig

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
# Result container
# ---------------------------------------------------------------------------


@dataclass
class InversionResult:
    """Container for tau-p inversion results and convergence history."""

    true_model: LayerModel
    initial_model: LayerModel
    recovered_model: LayerModel
    misfit_history: list[float] = field(default_factory=list)
    grad_norm_history: list[float] = field(default_factory=list)
    param_error_history: list[float] = field(default_factory=list)
    n_iterations: int = 0
    converged: bool = False


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
        R = kennett_reflectivity_torch(
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
            R = kennett_reflectivity_torch(
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
# Newton-LM solver
# ---------------------------------------------------------------------------


def newton_lm_step(
    g: np.ndarray,
    H: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Solve the damped Newton system ``(H + lambda I) dx = -g``.

    Args:
        g: Gradient vector, shape ``(n,)``.
        H: Hessian matrix, shape ``(n, n)``.
        lam: Levenberg-Marquardt damping parameter.

    Returns:
        Newton step dx, shape ``(n,)``.
    """
    n = len(g)
    return np.linalg.solve(H + lam * np.eye(n), -g)


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
    """Run a Newton-LM inversion in the tau-p domain.

    Generates synthetic observed data from *true_model*, creates a perturbed
    initial model, and iterates a full-Newton solver with Levenberg-Marquardt
    damping in log-parameter space until convergence.

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

    # Synthetic observed data at the true model
    R_obs: dict[float, torch.Tensor] = {}
    with torch.no_grad():
        for p_val in p_values:
            R_obs[p_val] = kennett_reflectivity_torch(
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
# Helpers
# ---------------------------------------------------------------------------

_LAYER_NAMES = ["Ocean", "Sediment", "Crust", "Upper mantle", "Half-space"]


def _tensors_to_layer_model(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    rho: torch.Tensor,
    thickness: torch.Tensor,
    ref_model: LayerModel,
) -> LayerModel:
    """Convert torch tensors back to a LayerModel (restore inf for half-space)."""
    h = thickness.detach().numpy().copy()
    h[h >= 1e29] = np.inf
    return LayerModel.from_arrays(
        alpha=alpha.detach().numpy(),
        beta=beta.detach().numpy(),
        rho=rho.detach().numpy(),
        thickness=h,
        Q_alpha=ref_model.Q_alpha,
        Q_beta=ref_model.Q_beta,
    )


def _depth_profile(
    model: LayerModel,
    param: str,
    halfspace_ext: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(depths, values)`` step-function arrays for *param*."""
    vals = getattr(model, param)
    depths: list[float] = []
    values: list[float] = []
    z = 0.0
    for i in range(model.n_layers):
        depths.append(z)
        values.append(float(vals[i]))
        h = float(model.thickness[i])
        if np.isinf(h):
            h = halfspace_ext
        z += h
        depths.append(z)
        values.append(float(vals[i]))
    return np.array(depths), np.array(values)


# ---------------------------------------------------------------------------
# Tau-p trace synthesis
# ---------------------------------------------------------------------------


def compute_taup_traces(
    model: LayerModel,
    p_values: list[float],
    T: float = 64.0,
    nw: int = 1024,
) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    """Compute Ricker-convolved tau-p seismograms at specified slownesses.

    Uses the NumPy Kennett reflectivity and the standard Ricker source
    (``source.ricker_spectrum``) followed by Hermitian-symmetric IFFT.

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
    from .kennett_reflectivity import kennett_reflectivity
    from .source import ricker_spectrum

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
        R = kennett_reflectivity(model, p_val, omega)
        Y = R * S
        Uwk = np.zeros(nt, dtype=np.complex128)
        Uwk[1:nw] = Y
        Uwk[nw + 1 :] = np.conj(Y[::-1])
        traces[p_val] = np.real(np.fft.fft(Uwk))

    return time, traces


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------


def write_model_table_latex(
    result: InversionResult,
    output: Path,
    layer_names: list[str] | None = None,
) -> None:
    r"""Write a LaTeX ``booktabs`` table comparing true/initial/recovered parameters.

    The generated ``.tex`` file can be included in a document via
    ``\input{<output>}``.  Requires the ``booktabs`` and ``siunitx``
    packages.

    Args:
        result: Inversion result.
        output: Path for the ``.tex`` file.
        layer_names: Custom layer names. Falls back to ``_LAYER_NAMES``.
    """
    names = layer_names if layer_names is not None else _LAYER_NAMES
    lines: list[str] = []
    a = lines.append

    a(r"\begin{table}[htbp]")
    a(r"\centering")
    a(r"\caption{True, initial, and recovered model parameters.}")
    a(r"\label{tab:taup-inversion-parameters}")
    a(r"\sisetup{round-mode=places, round-precision=3}")
    a(
        r"\begin{tabular}{l S[table-format=1.3] S[table-format=1.3]"
        r" S[table-format=1.3] S[table-format=1.3] r}"
    )
    a(r"\toprule")
    a(r"Layer & {$V_P$} & {$V_S$} & {$\rho$} & {$h$} & {Error} \\")
    a(r"      & {(km/s)} & {(km/s)} & {(g/cm$^3$)} & {(km)} & {(\%)} \\")
    a(r"\midrule")

    for tag, model in [
        ("True", result.true_model),
        ("Initial", result.initial_model),
        ("Recovered", result.recovered_model),
    ]:
        a(rf"\multicolumn{{6}}{{l}}{{\textit{{{tag} model}}}} \\")
        for i in range(model.n_layers):
            name = names[i] if i < len(names) else f"Layer {i}"
            vp = model.alpha[i]
            vs = model.beta[i]
            rho = model.rho[i]
            h = model.thickness[i]
            h_str = r"\infty" if np.isinf(h) else f"{h:.3f}"

            # Per-layer relative error (only for recovered)
            if tag == "Recovered":
                true = result.true_model
                diffs = [
                    abs(vp - true.alpha[i]) / max(true.alpha[i], 1e-30),
                    abs(rho - true.rho[i]) / max(true.rho[i], 1e-30),
                ]
                if true.beta[i] > 0:
                    diffs.append(
                        abs(vs - true.beta[i]) / max(true.beta[i], 1e-30),
                    )
                if not np.isinf(true.thickness[i]):
                    diffs.append(
                        abs(h - true.thickness[i]) / max(true.thickness[i], 1e-30),
                    )
                err_pct = max(diffs) * 100
                err_str = f"{err_pct:.1e}"
            else:
                err_str = ""

            a(
                rf"\quad {name} & {vp:.3f} & {vs:.3f} & {rho:.3f}"
                rf" & {h_str} & {err_str} \\"
            )
        if tag != "Recovered":
            a(r"\midrule")

    a(r"\bottomrule")
    a(r"\end{tabular}")
    a(r"\end{table}")

    output.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Matplotlib configuration for LaTeX-compatible PDF output
# ---------------------------------------------------------------------------

_LATEX_RCPARAMS: dict[str, object] = {
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.framealpha": 0.9,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.5,
}


# ---------------------------------------------------------------------------
# Publication figures
# ---------------------------------------------------------------------------


def _nice_ticks(v_min: float, v_max: float, n_target: int = 5) -> list[float]:
    """Compute aesthetically pleasing tick values for an axis range."""
    span = v_max - v_min
    if span <= 0:
        return [v_min]
    raw_step = span / n_target
    magnitude = 10 ** np.floor(np.log10(raw_step))
    residual = raw_step / magnitude
    if residual <= 1.5:
        nice_step = magnitude
    elif residual <= 3.5:
        nice_step = 2.0 * magnitude
    else:
        nice_step = 5.0 * magnitude
    start = np.ceil(v_min / nice_step) * nice_step
    ticks: list[float] = []
    v = start
    while v <= v_max + nice_step * 0.01:
        ticks.append(round(v, 10))
        v += nice_step
    return ticks


def _fmt_tick(v: float) -> str:
    """Format a tick value: integers without decimals, others with one."""
    return str(int(v)) if v == int(v) else f"{v:.1f}"


def write_model_profiles_tikz(result: InversionResult, output: Path) -> None:
    r"""Write a TikZ figure with 3-panel depth profiles (Vp, Vs, rho).

    Generates a ``\begin{figure}...\end{figure}`` environment with three
    side-by-side panels showing step-function velocity/density profiles
    for the true, initial, and recovered models.

    Requires ``\usepackage{tikz}`` in the document preamble.

    Args:
        result: Inversion result with true, initial, and recovered models.
        output: Path for the ``.tex`` file.
    """
    halfspace_ext = 2.0

    # Max depth across all three models
    max_depth = 0.0
    for model in [result.true_model, result.initial_model, result.recovered_model]:
        z = sum(float(h) for h in model.thickness if not np.isinf(h)) + halfspace_ext
        max_depth = max(max_depth, z)

    # Panel dimensions (cm)
    pw = 3.8
    ph = 7.0
    gap = 2.0
    y_scale = ph / max_depth

    params_info = [
        ("alpha", r"$V_P$ (km\,s$^{-1}$)"),
        ("beta", r"$V_S$ (km\,s$^{-1}$)"),
        ("rho", r"$\rho$ (g\,cm$^{-3}$)"),
    ]

    model_styles = [
        (result.true_model, "True", "black, line width=1.2pt"),
        (
            result.initial_model,
            "Initial",
            "blue!70!black, dashed, line width=0.8pt",
        ),
        (result.recovered_model, "Recovered", "red!70!black, line width=1.0pt"),
    ]

    out: list[str] = []
    a = out.append

    a(r"% Requires: \usepackage{tikz}")
    a(r"\begin{figure}[htbp]")
    a(r"\centering")
    a(r"\begin{tikzpicture}[>=stealth]")
    a("")

    for pi, (param, xlabel) in enumerate(params_info):
        x_off = pi * (pw + gap)

        # Value range across all models
        all_vals: list[float] = []
        for m, _, _ in model_styles:
            all_vals.extend(float(v) for v in getattr(m, param))
        v_lo, v_hi = min(all_vals), max(all_vals)
        pad = max(0.15 * (v_hi - v_lo), 0.2)
        v_min, v_max = v_lo - pad, v_hi + pad
        x_scale = pw / (v_max - v_min)

        a(f"  %% --- {param} ---")
        a(f"  \\begin{{scope}}[shift={{({x_off:.2f},0)}}]")

        # Frame
        a(f"    \\draw[thin] (0,0) rectangle ({pw:.2f},{-ph:.2f});")

        # Horizontal grid (depth)
        depth_ticks = list(range(0, int(max_depth) + 1))
        for dt in depth_ticks:
            yt = -dt * y_scale
            if abs(yt) > 0.01 and abs(yt + ph) > 0.01:
                a(
                    f"    \\draw[gray!25, very thin]"
                    f" (0,{yt:.3f}) -- ({pw:.2f},{yt:.3f});"
                )

        # Vertical grid + value ticks (top edge)
        vticks = _nice_ticks(v_min, v_max)
        for vt in vticks:
            xt = (vt - v_min) * x_scale
            if xt > 0.01 and abs(xt - pw) > 0.01:
                a(
                    f"    \\draw[gray!25, very thin]"
                    f" ({xt:.3f},0) -- ({xt:.3f},{-ph:.2f});"
                )
            a(f"    \\draw ({xt:.3f},0) -- ({xt:.3f},0.08);")
            a(
                f"    \\node[above, font=\\footnotesize]"
                f" at ({xt:.3f},0.08) {{{_fmt_tick(vt)}}};"
            )

        # Depth ticks (left side, first panel only)
        if pi == 0:
            for dt in depth_ticks:
                yt = -dt * y_scale
                a(f"    \\draw (0,{yt:.3f}) -- (-0.08,{yt:.3f});")
                a(f"    \\node[left, font=\\footnotesize] at (-0.1,{yt:.3f}) {{{dt}}};")

        # Panel label
        a(f"    \\node[above] at ({pw / 2:.2f},0.55) {{{xlabel}}};")

        # Step-function profiles
        for model, _label, style in model_styles:
            d, v = _depth_profile(model, param, halfspace_ext)
            coords: list[str] = []
            for di, vi in zip(d, v, strict=True):
                xt = (vi - v_min) * x_scale
                yt = -di * y_scale
                xt = max(0.0, min(float(pw), xt))
                yt = max(float(-ph), min(0.0, yt))
                coords.append(f"({xt:.4f},{yt:.4f})")
            a(f"    \\draw[{style}] {' -- '.join(coords)};")

        a("  \\end{scope}")
        a("")

    # Depth axis label
    a(f"  \\node[rotate=90] at (-1.0,{-ph / 2:.2f}) {{Depth (km)}};")

    # Legend (below panels, centred)
    total_w = 3 * pw + 2 * gap
    ly = -ph - 1.2
    lx = total_w / 2 - 3.5
    for i, (_, label, style) in enumerate(model_styles):
        xi = lx + i * 3.0
        a(
            f"  \\draw[{style}] ({xi:.2f},{ly:.2f})"
            f" -- ({xi + 0.8:.2f},{ly:.2f})"
            f" node[right, font=\\small] {{{label}}};"
        )

    a(r"\end{tikzpicture}")
    a(r"\caption{True, initial, and recovered model depth profiles.}")
    a(r"\label{fig:taup-depth-profiles}")
    a(r"\end{figure}")

    output.write_text("\n".join(out), encoding="utf-8")


def plot_taup_traces(
    result: InversionResult,
    output: Path,
    p_values: list[float] | None = None,
    T: float = 64.0,
    nw: int = 1024,
    t_max: float = 15.0,
) -> None:
    """Save a tau-p trace comparison figure.

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
    # Use the time array from the first call
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


def plot_convergence_curves(result: InversionResult, output: Path) -> None:
    """Save a 3-panel convergence figure (misfit, gradient, parameter error).

    Args:
        result: Inversion result with convergence histories.
        output: Output path (``.pdf`` recommended).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context(_LATEX_RCPARAMS):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        iters = np.arange(1, len(result.misfit_history) + 1)

        axes[0].semilogy(iters, result.misfit_history, "b-o", markersize=3)
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel(r"Misfit $\chi$")
        axes[0].set_title("Misfit convergence")
        axes[0].grid(True, alpha=0.3)

        axes[1].semilogy(
            iters,
            result.grad_norm_history,
            "r-o",
            markersize=3,
        )
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel(r"$\|\nabla\chi\|$")
        axes[1].set_title("Gradient norm")
        axes[1].grid(True, alpha=0.3)

        axes[2].semilogy(
            iters,
            result.param_error_history,
            "k-o",
            markersize=3,
        )
        axes[2].set_xlabel("Iteration")
        axes[2].set_ylabel(
            r"$\|\mathbf{m} - \mathbf{m}_\mathrm{true}\|"
            r" / \|\mathbf{m}_\mathrm{true}\|$",
        )
        axes[2].set_title("Relative parameter error")
        axes[2].grid(True, alpha=0.3)

        fig.tight_layout()
        plt.savefig(output, bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------


def _default_config() -> "InversionConfig":
    """Return a default InversionConfig matching the original demo."""
    from .inversion_config import InversionConfig, OutputConfig, TraceDisplayConfig

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
    """Run a tau-p inversion from a YAML config file or built-in defaults."""
    import argparse

    from .inversion_config import load_config, save_config

    parser = argparse.ArgumentParser(
        description="Tau-p inversion with Newton-Levenberg-Marquardt",
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
    logger.info("Running tau-p inversion...")
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
    save_config(cfg, outdir / "inversion_config.yaml")
    logger.info("Config saved -> %s", outdir / "inversion_config.yaml")

    formats = cfg.output.formats

    if "table" in formats:
        write_model_table_latex(
            result,
            outdir / "taup_model_parameters.tex",
            layer_names=cfg.layer_names,
        )
        logger.info("LaTeX table  -> %s", outdir / "taup_model_parameters.tex")

    if "profiles" in formats:
        write_model_profiles_tikz(result, outdir / "taup_model_profiles.tex")
        logger.info("Depth profiles -> %s", outdir / "taup_model_profiles.tex")

    if "traces" in formats:
        td = cfg.output.trace_display
        plot_taup_traces(
            result,
            outdir / "taup_trace_comparison.pdf",
            T=64.0,
            nw=td.nw,
            t_max=td.t_max,
        )
        logger.info("Trace comparison -> %s", outdir / "taup_trace_comparison.pdf")

    if "convergence" in formats:
        plot_convergence_curves(result, outdir / "taup_convergence.pdf")
        logger.info("Convergence -> %s", outdir / "taup_convergence.pdf")


if __name__ == "__main__":
    main()
