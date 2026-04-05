"""Investigate whether torch.func (vmap + jacrev/jacfwd) can speed up
the Jacobian computation for both GMM and Kennett.

torch.autograd.functional.jacobian loops over outputs, doing one backward
per output element. torch.func.jacrev with vmap can vectorize this.
torch.func.jacfwd uses forward-mode AD which loops over inputs instead —
better when n_params < nfreq.
"""

import time

import numpy as np
import torch
import torch.func

from GlobalMatrix.gmm_torch import gmm_reflectivity_torch
from Kennett_Reflectivity.kennett_seismogram import default_ocean_crust_model
from Kennett_Reflectivity.kennett_torch import (
    _pack_params,
    _unpack_params,
    kennett_reflectivity_torch,
    model_to_tensors,
)


def make_omega(nw: int) -> torch.Tensor:
    T = 64.0
    dw = 2.0 * np.pi / T
    return torch.arange(1, nw, dtype=torch.float64) * dw


def _timeit(fn, n_runs=3) -> float:
    """Median time in ms."""
    fn()  # warmup
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return np.median(times)


def make_forward_fn(reflectivity_fn, tensors, p, omega):
    """Create a function from params -> R for use with torch.func."""

    def forward(params):
        a, b, r, h = _unpack_params(
            params,
            tensors["alpha"],
            tensors["beta"],
            tensors["rho"],
            tensors["thickness"],
        )
        return reflectivity_fn(
            a,
            b,
            r,
            h,
            tensors["Q_alpha"],
            tensors["Q_beta"],
            p,
            omega,
        )

    return forward


def test_jacrev(tensors, omega, p=0.2):
    """Test torch.func.jacrev (vectorized reverse-mode)."""
    params = _pack_params(
        tensors["alpha"],
        tensors["beta"],
        tensors["rho"],
        tensors["thickness"],
    )

    for label, fn in [
        ("Kennett", kennett_reflectivity_torch),
        ("GMM", gmm_reflectivity_torch),
    ]:
        forward = make_forward_fn(fn, tensors, p, omega)

        # Real and imag parts
        def fwd_real(p_vec):
            return forward(p_vec).real

        def fwd_imag(p_vec):
            return forward(p_vec).imag

        try:
            t = _timeit(
                lambda: (
                    torch.func.jacrev(fwd_real)(params)
                    + 1j * torch.func.jacrev(fwd_imag)(params)
                )
            )
            J = torch.func.jacrev(fwd_real)(params) + 1j * torch.func.jacrev(fwd_imag)(
                params
            )
            print(f"  {label:8s} jacrev:  {t:>8.1f} ms  shape={tuple(J.shape)}")
        except Exception as e:
            print(f"  {label:8s} jacrev:  FAILED — {e}")


def test_jacfwd(tensors, omega, p=0.2):
    """Test torch.func.jacfwd (forward-mode AD)."""
    params = _pack_params(
        tensors["alpha"],
        tensors["beta"],
        tensors["rho"],
        tensors["thickness"],
    )

    for label, fn in [
        ("Kennett", kennett_reflectivity_torch),
        ("GMM", gmm_reflectivity_torch),
    ]:
        forward = make_forward_fn(fn, tensors, p, omega)

        def fwd_real(p_vec):
            return forward(p_vec).real

        def fwd_imag(p_vec):
            return forward(p_vec).imag

        try:
            t = _timeit(
                lambda: (
                    torch.func.jacfwd(fwd_real)(params)
                    + 1j * torch.func.jacfwd(fwd_imag)(params)
                )
            )
            J = torch.func.jacfwd(fwd_real)(params) + 1j * torch.func.jacfwd(fwd_imag)(
                params
            )
            print(f"  {label:8s} jacfwd:  {t:>8.1f} ms  shape={tuple(J.shape)}")
        except Exception as e:
            print(f"  {label:8s} jacfwd:  FAILED — {e}")


def test_old_jacobian(tensors, omega, p=0.2):
    """Baseline: torch.autograd.functional.jacobian."""
    params = _pack_params(
        tensors["alpha"],
        tensors["beta"],
        tensors["rho"],
        tensors["thickness"],
    )

    for label, fn in [
        ("Kennett", kennett_reflectivity_torch),
        ("GMM", gmm_reflectivity_torch),
    ]:
        forward = make_forward_fn(fn, tensors, p, omega)

        def fwd_real(p_vec):
            return forward(p_vec).real

        def fwd_imag(p_vec):
            return forward(p_vec).imag

        t = _timeit(
            lambda: (
                torch.autograd.functional.jacobian(fwd_real, params)
                + 1j * torch.autograd.functional.jacobian(fwd_imag, params)
            )
        )
        print(f"  {label:8s} autograd.functional.jacobian: {t:>8.1f} ms")


def test_vmap_jacrev(tensors, omega, p=0.2):
    """Test vmap + jacrev (vectorized reverse-mode over frequencies)."""
    params = _pack_params(
        tensors["alpha"],
        tensors["beta"],
        tensors["rho"],
        tensors["thickness"],
    )

    # Can we vmap over frequencies? Each freq is an independent solve.
    for label, fn in [
        ("Kennett", kennett_reflectivity_torch),
        ("GMM", gmm_reflectivity_torch),
    ]:
        # Create per-frequency function
        def per_freq_forward(params, w):
            a, b, r, h = _unpack_params(
                params,
                tensors["alpha"],
                tensors["beta"],
                tensors["rho"],
                tensors["thickness"],
            )
            R = fn(
                a,
                b,
                r,
                h,
                tensors["Q_alpha"],
                tensors["Q_beta"],
                p,
                w.unsqueeze(0),
            )
            return R[0]  # scalar complex

        def per_freq_real(params, w):
            return per_freq_forward(params, w).real

        def per_freq_imag(params, w):
            return per_freq_forward(params, w).imag

        try:
            # jacrev over params, vmap over frequencies
            def batched_jac(params):
                jr = torch.func.jacrev(per_freq_real, argnums=0)
                ji = torch.func.jacrev(per_freq_imag, argnums=0)
                J_real = torch.vmap(jr, in_dims=(None, 0))(params, omega)
                J_imag = torch.vmap(ji, in_dims=(None, 0))(params, omega)
                return J_real + 1j * J_imag

            t = _timeit(lambda: batched_jac(params))
            J = batched_jac(params)
            print(f"  {label:8s} vmap(jacrev): {t:>8.1f} ms  shape={tuple(J.shape)}")
        except Exception as e:
            print(f"  {label:8s} vmap(jacrev): FAILED — {type(e).__name__}: {e}")


def test_vmap_jacfwd(tensors, omega, p=0.2):
    """Test vmap + jacfwd (vectorized forward-mode over frequencies)."""
    params = _pack_params(
        tensors["alpha"],
        tensors["beta"],
        tensors["rho"],
        tensors["thickness"],
    )

    for label, fn in [
        ("Kennett", kennett_reflectivity_torch),
        ("GMM", gmm_reflectivity_torch),
    ]:

        def per_freq_forward(params, w):
            a, b, r, h = _unpack_params(
                params,
                tensors["alpha"],
                tensors["beta"],
                tensors["rho"],
                tensors["thickness"],
            )
            R = fn(
                a,
                b,
                r,
                h,
                tensors["Q_alpha"],
                tensors["Q_beta"],
                p,
                w.unsqueeze(0),
            )
            return R[0]

        def per_freq_real(params, w):
            return per_freq_forward(params, w).real

        def per_freq_imag(params, w):
            return per_freq_forward(params, w).imag

        try:

            def batched_jac(params):
                jf_r = torch.func.jacfwd(per_freq_real, argnums=0)
                jf_i = torch.func.jacfwd(per_freq_imag, argnums=0)
                J_real = torch.vmap(jf_r, in_dims=(None, 0))(params, omega)
                J_imag = torch.vmap(jf_i, in_dims=(None, 0))(params, omega)
                return J_real + 1j * J_imag

            t = _timeit(lambda: batched_jac(params))
            J = batched_jac(params)
            print(f"  {label:8s} vmap(jacfwd): {t:>8.1f} ms  shape={tuple(J.shape)}")
        except Exception as e:
            print(f"  {label:8s} vmap(jacfwd): FAILED — {type(e).__name__}: {e}")


def main():
    model = default_ocean_crust_model()
    tensors = model_to_tensors(model, requires_grad=True)

    print("=" * 72)
    print("Investigation: torch.func (vmap/jacrev/jacfwd) for GMM vs Kennett")
    print("=" * 72)

    for nw_label, nw in [("small (nfreq=15)", 16), ("medium (nfreq=63)", 64)]:
        omega = make_omega(nw)
        nfreq = nw - 1

        print(f"\n--- {nw_label}, n_params=15 ---")
        print(f"  (reverse-mode: {2 * nfreq} backward passes;")
        print(f"   forward-mode: {2 * 15} forward passes)")
        print()

        print("  Baseline (autograd.functional.jacobian — loops over outputs):")
        test_old_jacobian(tensors, omega)
        print()

        print("  torch.func.jacrev (vectorized reverse-mode):")
        test_jacrev(tensors, omega)
        print()

        print("  torch.func.jacfwd (forward-mode AD — loops over inputs):")
        test_jacfwd(tensors, omega)
        print()

        print("  vmap(jacrev) — vmap over frequencies, jacrev over params:")
        test_vmap_jacrev(tensors, omega)
        print()

        print("  vmap(jacfwd) — vmap over frequencies, jacfwd over params:")
        test_vmap_jacfwd(tensors, omega)
        print()

    print("=" * 72)


if __name__ == "__main__":
    main()
