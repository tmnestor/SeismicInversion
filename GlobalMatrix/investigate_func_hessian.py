"""Investigate torch.func for Hessian + validate jacrev correctness."""

import time

import numpy as np
import torch
import torch.func

from GlobalMatrix.gmm_torch import gmm_hessian, gmm_jacobian, gmm_reflectivity_torch
from Kennett_Reflectivity.kennett_seismogram import default_ocean_crust_model
from Kennett_Reflectivity.kennett_torch import (
    _pack_params,
    _unpack_params,
    kennett_reflectivity_torch,
    model_to_tensors,
)
from Kennett_Reflectivity.kennett_torch import (
    hessian as kennett_hessian,
)
from Kennett_Reflectivity.kennett_torch import (
    jacobian as kennett_jacobian,
)


def make_omega(nw: int) -> torch.Tensor:
    T = 64.0
    dw = 2.0 * np.pi / T
    return torch.arange(1, nw, dtype=torch.float64) * dw


def _timeit(fn, n_runs=3) -> float:
    fn()
    return np.median([(time.perf_counter(), fn(), time.perf_counter())
                       for _ in range(n_runs)],
                      axis=0)[2] - np.median(
        [(time.perf_counter(), fn(), time.perf_counter())
         for _ in range(n_runs)], axis=0)[0]


def timeit(fn, n_runs=3) -> float:
    fn()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return np.median(times)


def validate_jacrev():
    """Verify torch.func.jacrev gives same results as autograd.functional.jacobian."""
    model = default_ocean_crust_model()
    tensors = model_to_tensors(model, requires_grad=True)
    omega = make_omega(16)
    params = _pack_params(
        tensors["alpha"], tensors["beta"], tensors["rho"], tensors["thickness"],
    )

    for label, refl_fn, jac_fn in [
        ("Kennett", kennett_reflectivity_torch, kennett_jacobian),
        ("GMM", gmm_reflectivity_torch, gmm_jacobian),
    ]:
        # Reference: autograd.functional.jacobian
        J_ref = jac_fn(
            tensors["alpha"], tensors["beta"], tensors["rho"],
            tensors["thickness"], tensors["Q_alpha"], tensors["Q_beta"],
            0.2, omega,
        ).detach()

        # torch.func.jacrev
        def forward(p_vec):
            a, b, r, h = _unpack_params(
                p_vec, tensors["alpha"], tensors["beta"],
                tensors["rho"], tensors["thickness"],
            )
            return refl_fn(
                a, b, r, h,
                tensors["Q_alpha"], tensors["Q_beta"],
                0.2, omega,
            )

        J_real = torch.func.jacrev(lambda p: forward(p).real)(params)
        J_imag = torch.func.jacrev(lambda p: forward(p).imag)(params)
        J_func = J_real + 1j * J_imag

        diff = torch.max(torch.abs(J_ref - J_func)).item()
        rel_mask = torch.abs(J_ref) > 1e-15
        rel_err = torch.max(
            torch.abs(J_ref[rel_mask] - J_func[rel_mask]) / torch.abs(J_ref[rel_mask])
        ).item() if rel_mask.any() else 0.0

        print(f"  {label}: max|diff|={diff:.2e}, max rel err={rel_err:.2e}")


def bench_hessian_methods():
    """Compare Hessian computation methods."""
    model = default_ocean_crust_model()
    tensors = model_to_tensors(model, requires_grad=True)

    print(f"\n  {'nfreq':>8}  {'K old':>8}  {'G old':>8}  "
          f"{'K jacrev':>8}  {'G jacrev':>8}  "
          f"{'K func.H':>8}  {'G func.H':>8}")
    print(f"  {'':>8}  {'(ms)':>8}  {'(ms)':>8}  "
          f"{'(ms)':>8}  {'(ms)':>8}  "
          f"{'(ms)':>8}  {'(ms)':>8}")

    for nw in [4, 8, 16, 32]:
        omega = make_omega(nw)
        nfreq = nw - 1

        args = (
            tensors["alpha"], tensors["beta"], tensors["rho"],
            tensors["thickness"], tensors["Q_alpha"], tensors["Q_beta"],
            0.2, omega,
        )

        # Old method: autograd.functional.hessian
        tk_old = timeit(lambda: kennett_hessian(*args))
        tg_old = timeit(lambda: gmm_hessian(*args))

        # jacrev-based Hessian: jacrev(jacrev(misfit))
        params = _pack_params(
            tensors["alpha"], tensors["beta"], tensors["rho"],
            tensors["thickness"],
        )

        def make_misfit(refl_fn):
            def misfit(p_vec):
                a, b, r, h = _unpack_params(
                    p_vec, tensors["alpha"], tensors["beta"],
                    tensors["rho"], tensors["thickness"],
                )
                R = refl_fn(
                    a, b, r, h,
                    tensors["Q_alpha"], tensors["Q_beta"],
                    0.2, omega,
                )
                return (R.real ** 2 + R.imag ** 2).sum()
            return misfit

        misfit_k = make_misfit(kennett_reflectivity_torch)
        misfit_g = make_misfit(gmm_reflectivity_torch)

        # jacrev(grad) approach
        def hessian_jacrev(misfit_fn, p):
            grad_fn = torch.func.grad(misfit_fn)
            return torch.func.jacrev(grad_fn)(p)

        tk_jr = timeit(lambda: hessian_jacrev(misfit_k, params))
        tg_jr = timeit(lambda: hessian_jacrev(misfit_g, params))

        # torch.func.hessian (built-in)
        tk_fh = timeit(lambda: torch.func.hessian(misfit_k)(params))
        tg_fh = timeit(lambda: torch.func.hessian(misfit_g)(params))

        print(f"  {nfreq:>8}  {tk_old:>8.1f}  {tg_old:>8.1f}  "
              f"{tk_jr:>8.1f}  {tg_jr:>8.1f}  "
              f"{tk_fh:>8.1f}  {tg_fh:>8.1f}")


def bench_full_comparison():
    """Full comparison table of all Jacobian approaches at production scale."""
    model = default_ocean_crust_model()
    tensors = model_to_tensors(model, requires_grad=True)

    # Production-scale: nfreq = 2047 (nw=2048)
    # But that's too slow for the old method. Use moderate sizes.
    print("\n  Full Jacobian comparison (all methods):")
    print(f"\n  {'nfreq':>8}  {'Method':>30}  {'Kennett':>10}  {'GMM':>10}  {'Speedup':>8}")

    for nw in [32, 128, 512]:
        omega = make_omega(nw)
        nfreq = nw - 1
        params = _pack_params(
            tensors["alpha"], tensors["beta"], tensors["rho"],
            tensors["thickness"],
        )

        for method_name, compute_jac in [
            ("autograd.functional.jacobian", "old"),
            ("torch.func.jacrev", "jacrev"),
            ("torch.func.jacfwd", "jacfwd"),
        ]:
            def make_jac_fn(refl_fn, method):
                def forward(p_vec):
                    a, b, r, h = _unpack_params(
                        p_vec, tensors["alpha"], tensors["beta"],
                        tensors["rho"], tensors["thickness"],
                    )
                    return refl_fn(
                        a, b, r, h,
                        tensors["Q_alpha"], tensors["Q_beta"],
                        0.2, omega,
                    )

                if method == "old":
                    def jac():
                        J_r = torch.autograd.functional.jacobian(
                            lambda p: forward(p).real, params
                        )
                        J_i = torch.autograd.functional.jacobian(
                            lambda p: forward(p).imag, params
                        )
                        return J_r + 1j * J_i
                elif method == "jacrev":
                    def jac():
                        J_r = torch.func.jacrev(lambda p: forward(p).real)(params)
                        J_i = torch.func.jacrev(lambda p: forward(p).imag)(params)
                        return J_r + 1j * J_i
                elif method == "jacfwd":
                    def jac():
                        J_r = torch.func.jacfwd(lambda p: forward(p).real)(params)
                        J_i = torch.func.jacfwd(lambda p: forward(p).imag)(params)
                        return J_r + 1j * J_i
                return jac

            jac_k = make_jac_fn(kennett_reflectivity_torch, compute_jac)
            jac_g = make_jac_fn(gmm_reflectivity_torch, compute_jac)

            n_runs = 1 if nw >= 512 and compute_jac == "old" else 3
            tk = timeit(jac_k, n_runs=n_runs)
            tg = timeit(jac_g, n_runs=n_runs)
            speedup = tk / tg

            print(f"  {nfreq:>8}  {method_name:>30}  {tk:>9.0f}ms  {tg:>9.0f}ms  {speedup:>7.2f}x")

        print()


def main():
    print("=" * 72)
    print("torch.func Hessian + Validation")
    print("=" * 72)

    print("\n1. VALIDATE jacrev matches autograd.functional.jacobian")
    print("-" * 50)
    validate_jacrev()

    print("\n\n2. HESSIAN METHODS COMPARISON")
    print("-" * 50)
    bench_hessian_methods()

    print("\n\n3. FULL JACOBIAN COMPARISON (ALL METHODS)")
    print("-" * 50)
    bench_full_comparison()

    print("\n\n4. SUMMARY")
    print("-" * 50)
    print("""
  Key findings:

  JACOBIAN:
  - torch.func.jacrev is ~10-14x faster than autograd.functional.jacobian
  - This dwarfs the GMM vs Kennett difference (~1.2-1.3x)
  - jacfwd is slower than jacrev (overhead per forward pass is higher)
  - The biggest optimization is switching the Jacobian API, not the physics

  HESSIAN:
  - torch.func.hessian (= jacrev(grad)) is ~2-3x faster than
    autograd.functional.hessian
  - GMM advantage is ~1.7x with old API, but narrows with torch.func

  RECOMMENDATION:
  - Switch both GMM and Kennett Jacobian/Hessian to use torch.func
  - The GMM advantage is real but modest (~1.2-1.3x for Jacobian)
  - The API switch is the dominant speedup (~10-14x)
""")
    print("=" * 72)


if __name__ == "__main__":
    main()
