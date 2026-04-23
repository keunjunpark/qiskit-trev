"""Parameter-shift gradient benchmark: torch vs JAX.

plan/14 Step 4 **primary gate**: JAX gradient must be >= 2x torch at the
reference workload for us to pay for Step 5 TPU benchmarks. A 1.5x-2x
result stops the port at Step 4 — ship JAX as a GPU-side speedup, skip
TPU.

Usage:
    .venv/bin/python bench/bench_gradient.py
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

    model = TensorRingModel(qc, obs, rank=rank, device=device)
    return model, P


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
    params_np = (np.random.RandomState(0).rand(P) * 6.28).astype(np.float32)
    params_j = jnp.asarray(params_np)

    # first call compiles
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
    runs = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"

    workloads = [
        dict(N=6,  reps=1, rank=6,  C=10),
        dict(N=8,  reps=1, rank=8,  C=20),
        dict(N=10, reps=2, rank=10, C=50),
    ]

    for w in workloads:
        N, reps, rank, C = w["N"], w["reps"], w["rank"], w["C"]
        model, P = build_model(N, reps, rank, C, device)
        print(f"\n=== N={N} reps={reps} rank={rank} obs_terms={C} P={P} ===")

        ts_t = time_torch(model, P, runs)
        print(f"torch : {fmt(ts_t)}")
        ts_j, compile_s = time_jax(model, P, runs)
        print(f"jax   : {fmt(ts_j)}  (first-call compile={compile_s*1e3:.1f} ms)")
        ratio = statistics.median(ts_t) / statistics.median(ts_j)
        verdict = "PASS >=2x" if ratio >= 2.0 else ("Marginal 1.5x-2x" if ratio >= 1.5 else "Below gate")
        print(f"→ torch/jax = {ratio:.2f}x  [{verdict}]")


if __name__ == "__main__":
    main()
