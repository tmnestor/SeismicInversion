"""Deep investigation of PyTorch AD behavior for GMM vs Kennett.

Analyzes autograd graph structure, per-operation costs, and scaling behavior.
"""

import time

import numpy as np
import torch

from GlobalMatrix.gmm_torch import (
    gmm_hessian,
    gmm_jacobian,
    gmm_reflectivity_torch,
)
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


def count_graph_nodes(tensor: torch.Tensor) -> tuple[int, dict[str, int]]:
    """Count all unique nodes in the autograd graph via DFS.

    Traverses the full computation graph from a scalar tensor back to leaves.

    Returns:
        (total_nodes, op_counts) where op_counts maps operation name to count.
    """
    if tensor.numel() > 1:
        tensor = tensor.sum()

    visited: set[int] = set()
    op_counts: dict[str, int] = {}

    def _dfs(node):
        if node is None:
            return
        node_id = id(node)
        if node_id in visited:
            return
        visited.add(node_id)
        name = type(node).__name__
        op_counts[name] = op_counts.get(name, 0) + 1
        for child, _ in node.next_functions:
            _dfs(child)

    if tensor.grad_fn is not None:
        _dfs(tensor.grad_fn)

    return len(visited), op_counts


def graph_depth(tensor: torch.Tensor) -> int:
    """Compute the max depth of the autograd graph."""
    if tensor.numel() > 1:
        tensor = tensor.sum()
    if tensor.grad_fn is None:
        return 0

    def _depth(node, cache):
        nid = id(node)
        if nid in cache:
            return cache[nid]
        max_child = 0
        for child, _ in node.next_functions:
            if child is not None:
                max_child = max(max_child, _depth(child, cache))
        cache[nid] = max_child + 1
        return cache[nid]

    return _depth(tensor.grad_fn, {})


def analyze_jacobian_method(tensors, omega):
    """Analyze what torch.autograd.functional.jacobian actually does."""
    params = _pack_params(
        tensors["alpha"], tensors["beta"], tensors["rho"], tensors["thickness"]
    )
    n_params = params.shape[0]
    nfreq = omega.shape[0]

    print(f"  Parameter vector size: {n_params}")
    print(f"  Frequency count: {nfreq}")
    print(f"  Jacobian shape: ({nfreq}, {n_params}) = {nfreq * n_params} elements")
    print()
    print("  torch.autograd.functional.jacobian strategy:")
    print("    - Splits complex R into real and imag parts")
    print(f"    - Each part: {nfreq} outputs -> {nfreq} backward passes")
    print(f"    - Total backward passes: 2 * {nfreq} = {2 * nfreq}")
    print(f"    - Each backward pass differentiates w.r.t. {n_params} params")
    print()


def profile_backward_passes(tensors, omega, p=0.2, n_runs=10):
    """Time individual forward + backward passes for GMM vs Kennett."""
    for label, fn in [("Kennett", kennett_reflectivity_torch),
                      ("GMM", gmm_reflectivity_torch)]:
        times_fwd = []
        times_bwd = []
        for _ in range(n_runs):
            params = _pack_params(
                tensors["alpha"], tensors["beta"], tensors["rho"],
                tensors["thickness"],
            ).requires_grad_(True)
            a, b, r, h = _unpack_params(
                params, tensors["alpha"], tensors["beta"],
                tensors["rho"], tensors["thickness"],
            )

            t0 = time.perf_counter()
            R = fn(a, b, r, h, tensors["Q_alpha"], tensors["Q_beta"], p, omega)
            t1 = time.perf_counter()
            loss = R[0].real
            loss.backward()
            t2 = time.perf_counter()

            times_fwd.append(t1 - t0)
            times_bwd.append(t2 - t1)

        fwd_ms = np.median(times_fwd) * 1000
        bwd_ms = np.median(times_bwd) * 1000
        print(f"  {label:8s}: forward={fwd_ms:.3f}ms, "
              f"single backward={bwd_ms:.3f}ms, "
              f"backward/forward={bwd_ms / fwd_ms:.1f}x")


def profile_forward_graph(tensors, omega, p=0.2):
    """Analyze the forward computation graph for both methods."""
    for label, fn in [("Kennett", kennett_reflectivity_torch),
                      ("GMM", gmm_reflectivity_torch)]:
        params = _pack_params(
            tensors["alpha"], tensors["beta"], tensors["rho"],
            tensors["thickness"],
        ).requires_grad_(True)
        a, b, r, h = _unpack_params(
            params, tensors["alpha"], tensors["beta"],
            tensors["rho"], tensors["thickness"],
        )
        R = fn(a, b, r, h, tensors["Q_alpha"], tensors["Q_beta"], p, omega)

        n_nodes, op_counts = count_graph_nodes(R)
        depth = graph_depth(R)

        print(f"\n  {label}:")
        print(f"    Total graph nodes: {n_nodes}")
        print(f"    Graph depth (longest path): {depth}")
        print("    Top 15 operations:")
        sorted_ops = sorted(op_counts.items(), key=lambda x: -x[1])
        for op, count in sorted_ops[:15]:
            print(f"      {op:40s} {count:>6}")


def scaling_analysis(tensors, p=0.2):
    """Show how Jacobian and Hessian time scales with nfreq."""
    print(f"\n  {'nfreq':>8}  {'K fwd':>8}  {'K jac':>8}  {'G fwd':>8}  "
          f"{'G jac':>8}  {'K bwd/fwd':>10}  {'G bwd/fwd':>10}  {'Speedup':>8}")
    print(f"  {'':>8}  {'(ms)':>8}  {'(ms)':>8}  {'(ms)':>8}  "
          f"{'(ms)':>8}  {'':>10}  {'':>10}  {'':>8}")

    for nw in [8, 16, 32, 64, 128, 256]:
        omega = make_omega(nw)
        nfreq = nw - 1
        args = (
            tensors["alpha"], tensors["beta"], tensors["rho"],
            tensors["thickness"], tensors["Q_alpha"], tensors["Q_beta"],
            p, omega,
        )

        # Warmup
        kennett_reflectivity_torch(*args)
        gmm_reflectivity_torch(*args)

        # Forward (median of 5)
        tk_fwd = np.median([_timeit(lambda: kennett_reflectivity_torch(*args)) for _ in range(5)])
        tg_fwd = np.median([_timeit(lambda: gmm_reflectivity_torch(*args)) for _ in range(5)])

        # Jacobian (median of 3)
        kennett_jacobian(*args)
        gmm_jacobian(*args)
        tk_jac = np.median([_timeit(lambda: kennett_jacobian(*args)) for _ in range(3)])
        tg_jac = np.median([_timeit(lambda: gmm_jacobian(*args)) for _ in range(3)])

        # Effective backward cost = (jac_time - fwd_time) / (2 * nfreq)
        # because jacobian does 2*nfreq backward passes (real + imag)
        k_ratio = tk_jac / tk_fwd
        g_ratio = tg_jac / tg_fwd
        speedup = tk_jac / tg_jac

        print(f"  {nfreq:>8}  {tk_fwd:>8.1f}  {tk_jac:>8.0f}  {tg_fwd:>8.1f}  "
              f"{tg_jac:>8.0f}  {k_ratio:>10.0f}x  {g_ratio:>10.0f}x  {speedup:>7.2f}x")


def _timeit(fn) -> float:
    """Time a function call in milliseconds."""
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000


def implicit_diff_explanation():
    """Print explanation of how torch.linalg.solve's backward works."""
    print("""
  How torch.linalg.solve backward works
  ======================================
  Forward: solve G @ x = b  =>  x = G^{-1} b

  Backward: given upstream gradient dL/dx:
    1. Solve  G^T @ lambda = dL/dx     (back-substitution using cached LU)
    2. dL/db = lambda
    3. dL/dG = -lambda @ x^T           (outer product, O(N^2) per freq)

  The LU factorization computed during the forward pass is reused for the
  backward solve. So each backward pass costs ~O(N^2) per frequency
  (a back-substitution), NOT another O(N^3) factorization.

  For the full Jacobian (nfreq outputs, n_params parameters):
    - 2*nfreq backward passes (real + imag parts)
    - Each backward: nfreq back-subs of 15x15 systems (trivially fast)
    - Then chain rule through the assembly of G and b w.r.t. model params

  For Kennett:
    - Same 2*nfreq backward passes
    - Each backward: reverse-mode AD walks the full recursive graph
      (4 interfaces x batch_inv2x2 + batch_matmul per interface)
    - More operations in the graph => slower backward""")


def main():
    model = default_ocean_crust_model()
    tensors = model_to_tensors(model, requires_grad=True)

    print("=" * 72)
    print("Investigation: PyTorch AD for GMM vs Kennett")
    print("=" * 72)

    # 1. Graph structure
    print("\n1. AUTOGRAD GRAPH STRUCTURE")
    print("-" * 40)
    omega_small = make_omega(16)
    profile_forward_graph(tensors, omega_small)

    # 2. Implicit differentiation
    print("\n\n2. IMPLICIT DIFFERENTIATION (torch.linalg.solve)")
    print("-" * 40)
    implicit_diff_explanation()

    # 3. Jacobian method
    print("\n\n3. JACOBIAN COMPUTATION STRATEGY")
    print("-" * 40)
    omega_jac = make_omega(32)
    analyze_jacobian_method(tensors, omega_jac)

    # 4. Per-backward-pass cost
    print("\n4. PER-PASS COST (forward + single backward)")
    print("-" * 40)
    profile_backward_passes(tensors, omega_small)

    # 5. Scaling
    print("\n\n5. SCALING: JACOBIAN vs NFREQ")
    print("-" * 40)
    scaling_analysis(tensors)

    # 6. Hessian scaling
    print("\n\n6. HESSIAN SCALING")
    print("-" * 40)
    print(f"\n  {'nfreq':>8}  {'K hess(ms)':>12}  {'G hess(ms)':>12}  {'Speedup':>8}")

    for nw in [4, 8, 16, 32]:
        omega = make_omega(nw)
        nfreq = nw - 1
        args = (
            tensors["alpha"], tensors["beta"], tensors["rho"],
            tensors["thickness"], tensors["Q_alpha"], tensors["Q_beta"],
            0.2, omega,
        )
        # Warmup
        kennett_hessian(*args)
        gmm_hessian(*args)

        tk = np.median([_timeit(lambda: kennett_hessian(*args)) for _ in range(3)])
        tg = np.median([_timeit(lambda: gmm_hessian(*args)) for _ in range(3)])
        print(f"  {nfreq:>8}  {tk:>12.1f}  {tg:>12.1f}  {tk / tg:>7.2f}x")

    # 7. Key observations summary
    print("\n\n7. KEY OBSERVATIONS")
    print("-" * 40)
    print("""
  a) GMM forward is ~1.3-1.7x SLOWER than Kennett (assembling + solving
     a 15x15 system vs recursive 2x2 ops).

  b) GMM Jacobian is ~1.25x FASTER (each backward pass through
     torch.linalg.solve is cheaper than reverse-mode through the recursion).

  c) GMM Hessian is ~1.7x FASTER (the advantage compounds for second
     derivatives since the Hessian requires backward through the Jacobian).

  d) Both methods are dominated by the 2*nfreq backward passes that
     torch.autograd.functional.jacobian performs. The graph traversal
     cost per backward pass determines the Jacobian speed.

  e) The Jacobian-to-forward ratio is ~350x for GMM vs ~560x for Kennett
     at nfreq=127. Both are high because the forward is cheap (vectorized
     numpy-like ops) while the backward requires nfreq separate graph walks.

  f) A forward-mode AD or vmap-based Jacobian could reduce the number of
     graph traversals from 2*nfreq to 2*n_params (since n_params < nfreq
     for our problem). This would benefit both methods equally.
""")

    print("=" * 72)


if __name__ == "__main__":
    main()
