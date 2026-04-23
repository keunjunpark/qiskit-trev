"""Gradient benchmark at larger bond dim — TPU crossover test.

Plan/14 predicted TPU starts winning at χ ≥ 32 where the MXU (128×128) is
better-utilised. v5e loses at small χ (see bench_gradient.py results);
this bench sweeps χ from small to large on whatever hardware you run it on.

Workloads are sized so the HBM footprint fits both v5e (15.75 GB) and
consumer 16 GB GPUs. Sparse Hamiltonian (C small) and moderate N (≤ 8)
keep intermediates bounded.

Usage:
    python bench/bench_gradient_large_chi.py
"""

from __future__ import annotations

import math
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qiskit_trev.gradient import BatchParameterShiftGradient
from qiskit_trev.model import TensorRingModel


def build_model(N: int, reps: int, rank: int, num_obs_terms: int, device: str):
    qc = QuantumCircuit(N)
    P = 0
    for _ in range(reps):
        for q in range(N):
            qc.ry(0.0, q)
            P += 1
        for q in range(N):
            qc.cx(q, (q + 1) % N)

    rng = np.random.RandomState(0)
    terms = []
    for _ in range(num_obs_terms):
        s = "".join(rng.choice(["I", "Z"]) for _ in range(N))
        if "Z" not in s:
            s = "Z" + s[1:]
        terms.append((s, float(rng.rand() - 0.5)))
    obs = SparsePauliOp.from_list(terms)
    return TensorRingModel(qc, obs, rank=rank, device=device), P


def time_torch(model, P: int, runs: int):
    grad_fn = BatchParameterShiftGradient(model)
    device = torch.device(model.device_str)
    params = (torch.rand(P, generator=torch.Generator().manual_seed(0)) * 6.28).to(device)
    for _ in range(2):
        grad_fn(params)
        if device.type == "cuda":
            torch.cuda.synchronize()
    ts = []
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        grad_fn(params)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return ts


def time_jax(model, P: int, runs: int):
    grad_fn = BatchParameterShiftGradient(model)
    params_j = jnp.asarray(
        (np.random.RandomState(0).rand(P) * 6.28).astype(np.float32)
    )
    t0 = time.perf_counter()
    grad_fn(params_j)
    compile_s = time.perf_counter() - t0
    for _ in range(2):
        grad_fn(params_j)
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        grad_fn(params_j)
        ts.append(time.perf_counter() - t0)
    return ts, compile_s


def fmt(ts):
    return (
        f"median={statistics.median(ts)*1e3:9.2f} ms  "
        f"min={min(ts)*1e3:9.2f} ms  "
        f"stdev={statistics.stdev(ts)*1e3:7.2f} ms"
    )


def main():
    torch.set_num_threads(1)
    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"
    runs = 5 if has_cuda else 3

    # χ sweep at fixed N=6, reps=1, C=5. P = N * reps = 6; 2P state batch = 12.
    # HBM per chi^4 intermediate (fp32): chi=16 → 0.5 MB, chi=32 → 8 MB,
    # chi=48 → 42 MB, chi=64 → 134 MB. With C=5 and 2P=12 outer batch,
    # even chi=64 stays well under 16 GB.
    chi_sweep = [6, 12, 16, 24, 32, 48]
    if has_cuda:
        chi_sweep.append(64)

    print(f"device = {device} | torch.cuda.is_available = {has_cuda}")

    for chi in chi_sweep:
        print(f"\n=== N=6 reps=1 rank={chi} obs_terms=5 P=6 ===")
        try:
            model, P = build_model(N=6, reps=1, rank=chi, num_obs_terms=5,
                                   device=device)
        except Exception as e:
            print(f"skip (model build failed): {type(e).__name__}: {e}")
            continue

        try:
            ts_t = time_torch(model, P, runs)
            print(f"torch : {fmt(ts_t)}")
            t_med = statistics.median(ts_t)
        except Exception as e:
            print(f"torch failed: {type(e).__name__}: {str(e)[:200]}")
            t_med = None

        try:
            ts_j, compile_s = time_jax(model, P, runs)
            print(f"jax   : {fmt(ts_j)}  (first-call compile={compile_s*1e3:.1f} ms)")
            j_med = statistics.median(ts_j)
        except Exception as e:
            print(f"jax failed: {type(e).__name__}: {str(e)[:200]}")
            j_med = None

        if t_med and j_med:
            ratio = t_med / j_med
            verdict = "jax faster" if ratio > 1 else "torch faster"
            print(f"→ torch/jax = {ratio:.2f}x  ({verdict})")


if __name__ == "__main__":
    main()
