"""Full training wall-time bench — does JAX first-call compile dominate?

plan/14 bench_gradient.py reports per-step medians after warm-up. That
hides the first-call compile. This bench times a full N-iteration
gradient-descent loop including compile and reports total seconds, so you
see the *realistic* wall clock a user would experience.

Run on GPU — numbers on CPU/TPU aren't what the decision needs.

Usage:
    python bench/bench_training_wall_time.py
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import torch
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qiskit_trev.gradient import BatchParameterShiftGradient
from qiskit_trev.model import TensorRingModel


def build(N: int, reps: int, rank: int, C: int, device: str):
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
    for _ in range(C):
        s = "".join(rng.choice(["I", "Z"]) for _ in range(N))
        if "Z" not in s:
            s = "Z" + s[1:]
        terms.append((s, float(rng.rand() - 0.5)))
    obs = SparsePauliOp.from_list(terms)
    return TensorRingModel(qc, obs, rank=rank, device=device), P


def train_torch(model, P: int, iters: int, lr: float = 0.01) -> float:
    grad_fn = BatchParameterShiftGradient(model)
    device = torch.device(model.device_str)
    params = (
        torch.rand(P, generator=torch.Generator().manual_seed(0)) * 6.28
    ).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        g = grad_fn(params).to(device=device, dtype=params.dtype)
        params = params - lr * g
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def train_jax(model, P: int, iters: int, lr: float = 0.01) -> float:
    grad_fn = BatchParameterShiftGradient(model)
    params = jnp.asarray(
        (np.random.RandomState(0).rand(P) * 6.28).astype(np.float32)
    )

    t0 = time.perf_counter()
    for _ in range(iters):
        g = grad_fn(params)
        params = params - lr * g
    params.block_until_ready()
    return time.perf_counter() - t0


def main():
    torch.set_num_threads(1)
    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"

    workloads = [
        dict(N=6,  reps=1, rank=6,  C=10),
        dict(N=8,  reps=1, rank=8,  C=20),
        dict(N=10, reps=2, rank=10, C=50),
    ]
    iter_counts = [1, 10, 100, 500]

    print(f"device = {device}")
    print(
        f"{'N':>3} {'reps':>4} {'rank':>5} {'C':>4} "
        f"{'iters':>6} {'torch (s)':>10} {'jax (s)':>10} "
        f"{'torch/jax':>10} {'jax/iter (ms)':>14}"
    )

    for w in workloads:
        N, reps, rank, C = w["N"], w["reps"], w["rank"], w["C"]
        for iters in iter_counts:
            model, P = build(N, reps, rank, C, device)
            try:
                t_t = train_torch(model, P, iters)
            except Exception as e:
                t_t = float("nan")
                print(f"torch failed @ iters={iters}: {type(e).__name__}: {e}")
            try:
                t_j = train_jax(model, P, iters)
            except Exception as e:
                t_j = float("nan")
                print(f"jax failed @ iters={iters}: {type(e).__name__}: {e}")

            ratio = (t_t / t_j) if (t_j and t_j > 0 and t_t == t_t) else float("nan")
            per_iter_ms = (t_j / iters * 1e3) if (t_j and t_j > 0) else float("nan")
            print(
                f"{N:>3} {reps:>4} {rank:>5} {C:>4} "
                f"{iters:>6} {t_t:>10.2f} {t_j:>10.2f} "
                f"{ratio:>9.2f}x {per_iter_ms:>14.3f}"
            )


if __name__ == "__main__":
    main()
