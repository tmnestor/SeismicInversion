"""Benchmark PyTorch AD for GMM vs Kennett Jacobian and Hessian.

Profiles wall-clock time and memory for different frequency counts.
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
    hessian as kennett_hessian,
)
from Kennett_Reflectivity.kennett_torch import (
    jacobian as kennett_jacobian,
)
from Kennett_Reflectivity.kennett_torch import (
    kennett_reflectivity_torch,
    model_to_tensors,
)


def make_omega(nw: int) -> torch.Tensor:
    T = 64.0
    dw = 2.0 * np.pi / T
    return torch.arange(1, nw, dtype=torch.float64) * dw


def bench_forward(tensors, omega, p=0.2, n_runs=5):
    """Benchmark forward pass."""
    args = (
        tensors["alpha"],
        tensors["beta"],
        tensors["rho"],
        tensors["thickness"],
        tensors["Q_alpha"],
        tensors["Q_beta"],
        p,
        omega,
    )

    # Warmup
    kennett_reflectivity_torch(*args)
    gmm_reflectivity_torch(*args)

    times_k = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        kennett_reflectivity_torch(*args)
        times_k.append(time.perf_counter() - t0)

    times_g = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        gmm_reflectivity_torch(*args)
        times_g.append(time.perf_counter() - t0)

    return np.median(times_k), np.median(times_g)


def bench_jacobian(tensors, omega, p=0.2, n_runs=3):
    """Benchmark Jacobian computation."""
    args = (
        tensors["alpha"],
        tensors["beta"],
        tensors["rho"],
        tensors["thickness"],
        tensors["Q_alpha"],
        tensors["Q_beta"],
        p,
        omega,
    )

    # Warmup
    kennett_jacobian(*args)
    gmm_jacobian(*args)

    times_k = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        J_k = kennett_jacobian(*args)
        times_k.append(time.perf_counter() - t0)

    times_g = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        J_g = gmm_jacobian(*args)
        times_g.append(time.perf_counter() - t0)

    return np.median(times_k), np.median(times_g), J_k, J_g


def bench_hessian(tensors, omega, p=0.2, n_runs=3):
    """Benchmark Hessian computation."""
    args = (
        tensors["alpha"],
        tensors["beta"],
        tensors["rho"],
        tensors["thickness"],
        tensors["Q_alpha"],
        tensors["Q_beta"],
        p,
        omega,
    )

    # Warmup
    kennett_hessian(*args)
    gmm_hessian(*args)

    times_k = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        H_k = kennett_hessian(*args)
        times_k.append(time.perf_counter() - t0)

    times_g = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        H_g = gmm_hessian(*args)
        times_g.append(time.perf_counter() - t0)

    return np.median(times_k), np.median(times_g), H_k, H_g


def main():
    model = default_ocean_crust_model()
    tensors = model_to_tensors(model, requires_grad=True)

    print("=" * 72)
    print("GMM vs Kennett — PyTorch AD Benchmark")
    print("=" * 72)
    print(f"Model: {model.n_layers} layers, 15x15 GMM system")
    print()

    # --- Forward pass ---
    print("--- Forward Pass (median of 5 runs) ---")
    print(f"{'nfreq':>8}  {'Kennett (ms)':>14}  {'GMM (ms)':>14}  {'Ratio':>8}")
    for nw in [16, 32, 64, 128, 256, 512, 1024]:
        omega = make_omega(nw)
        tk, tg = bench_forward(tensors, omega)
        ratio = tg / tk if tk > 0 else float("inf")
        print(f"{nw - 1:>8}  {tk * 1000:>14.2f}  {tg * 1000:>14.2f}  {ratio:>8.2f}x")
    print()

    # --- Jacobian ---
    print("--- Jacobian (median of 3 runs) ---")
    print(
        f"{'nfreq':>8}  {'Kennett (ms)':>14}  {'GMM (ms)':>14}  {'Ratio':>8}  {'Max |diff|':>12}"
    )
    for nw in [8, 16, 32, 64, 128]:
        omega = make_omega(nw)
        tk, tg, J_k, J_g = bench_jacobian(tensors, omega)
        diff = torch.max(torch.abs(J_k - J_g)).item()
        ratio = tg / tk if tk > 0 else float("inf")
        print(
            f"{nw - 1:>8}  {tk * 1000:>14.2f}  {tg * 1000:>14.2f}  {ratio:>8.2f}x  {diff:>12.2e}"
        )
    print()

    # --- Hessian ---
    print("--- Hessian (median of 3 runs) ---")
    print(
        f"{'nfreq':>8}  {'Kennett (ms)':>14}  {'GMM (ms)':>14}  {'Ratio':>8}  {'Max |diff|':>12}"
    )
    for nw in [4, 8, 16, 32]:
        omega = make_omega(nw)
        tk, tg, H_k, H_g = bench_hessian(tensors, omega)
        diff = torch.max(torch.abs(H_k - H_g)).item()
        ratio = tg / tk if tk > 0 else float("inf")
        print(
            f"{nw - 1:>8}  {tk * 1000:>14.2f}  {tg * 1000:>14.2f}  {ratio:>8.2f}x  {diff:>12.2e}"
        )
    print()

    # --- Autograd graph analysis ---
    print("--- Autograd Graph Depth Analysis ---")
    omega_small = make_omega(8)
    args = (
        tensors["alpha"],
        tensors["beta"],
        tensors["rho"],
        tensors["thickness"],
        tensors["Q_alpha"],
        tensors["Q_beta"],
        0.2,
        omega_small,
    )

    # Count grad_fn nodes for Kennett
    R_k = kennett_reflectivity_torch(*args)
    k_nodes = _count_graph_nodes(R_k)

    R_g = gmm_reflectivity_torch(*args)
    g_nodes = _count_graph_nodes(R_g)

    print(f"Kennett autograd graph nodes: {k_nodes}")
    print(f"GMM autograd graph nodes:     {g_nodes}")
    print()

    # --- Memory analysis ---
    print("--- Tensor allocation in forward pass ---")
    for label, fn in [
        ("Kennett", kennett_reflectivity_torch),
        ("GMM", gmm_reflectivity_torch),
    ]:
        omega_mem = make_omega(256)
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

        # Count tensors before and after
        import gc

        gc.collect()
        before = len([obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor)])
        R = fn(
            tensors["alpha"],
            tensors["beta"],
            tensors["rho"],
            tensors["thickness"],
            tensors["Q_alpha"],
            tensors["Q_beta"],
            0.2,
            omega_mem,
        )
        after = len([obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor)])
        print(f"{label}: ~{after - before} new tensors created during forward pass")

    print()
    print("=" * 72)


def _count_graph_nodes(tensor: torch.Tensor) -> int:
    """Count unique nodes in the autograd graph."""
    visited = set()
    queue = []
    if tensor.grad_fn is not None:
        queue.append(tensor.grad_fn)
    while queue:
        node = queue.pop()
        if id(node) in visited:
            continue
        visited.add(id(node))
        for child, _ in node.next_functions:
            if child is not None:
                queue.append(child)
    return len(visited)


if __name__ == "__main__":
    main()
