"""End-to-end forward benchmark: torch vs JAX for build_batch + expectation.

plan/14 Step 3. Not a gating benchmark — per the plan, "forward time on
JAX/GPU >= torch" is sufficient to continue to Step 4. The real speed
target is Step 4's vmap'd parameter-shift gradient.

Usage:
    .venv/bin/python bench/bench_forward.py
"""

from __future__ import annotations

import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch

from qiskit_trev.hamiltonian import Hamiltonian
from qiskit_trev.measure.efficient_contraction import batched_expectation_value
from qiskit_trev.tensor_ring.state import GateInstruction, TensorRingState


def _ry_cnot_circuit(N: int, reps: int):
    gates: list[GateInstruction] = []
    p = 0
    for _ in range(reps):
        for q in range(N):
            gates.append(GateInstruction("RY", (q,), params=(0.0,), param_indices=(p,)))
            p += 1
        for q in range(N):
            gates.append(GateInstruction("CNOT", (q, (q + 1) % N)))
    return gates, p


def _ham(N: int, C: int, seed: int = 1) -> Hamiltonian:
    rng = np.random.RandomState(seed)
    terms = []
    for _ in range(C):
        s = "".join(rng.choice(["I", "Z"]) for _ in range(N))
        terms.append((s, float(rng.rand() - 0.5)))
    if "Z" not in terms[0][0]:
        terms[0] = ("Z" + terms[0][0][1:], terms[0][1])
    return Hamiltonian.from_pauli_list(terms)


def fmt(ts):
    return (
        f"median={statistics.median(ts)*1e3:8.2f} ms  "
        f"min={min(ts)*1e3:8.2f} ms  "
        f"stdev={statistics.stdev(ts)*1e3:6.2f} ms"
    )


def time_torch(N, chi, B, reps, ham, runs):
    gates, P = _ry_cnot_circuit(N, reps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = (torch.rand(B, P, generator=torch.Generator().manual_seed(0)) * 6.28).to(device)
    trs = TensorRingState(num_qubits=N, rank=chi, device=str(device))

    # warm-up
    for _ in range(3):
        state = trs.build_batch(gates, params)
        batched_expectation_value(state, ham)
        if device.type == "cuda":
            torch.cuda.synchronize()

    ts = []
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        state = trs.build_batch(list(gates), params)
        _ = batched_expectation_value(state, ham)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return ts


def time_jax(N, chi, B, reps, ham, runs):
    gates, P = _ry_cnot_circuit(N, reps)
    params_t = torch.rand(B, P, generator=torch.Generator().manual_seed(0)) * 6.28
    params_j = jnp.asarray(params_t.numpy())
    trs = TensorRingState(num_qubits=N, rank=chi)

    # warm-up
    for _ in range(3):
        state = trs.build_batch(list(gates), params_j)
        _ = batched_expectation_value(state, ham).block_until_ready()

    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        state = trs.build_batch(list(gates), params_j)
        _ = batched_expectation_value(state, ham).block_until_ready()
        ts.append(time.perf_counter() - t0)
    return ts


def main():
    torch.set_num_threads(1)
    runs = 10

    for N, chi, B, reps, C in [(8, 8, 16, 2, 10), (10, 12, 16, 2, 20), (12, 12, 32, 2, 50)]:
        ham = _ham(N, C)
        print(f"\n=== N={N} chi={chi} B={B} reps={reps} C={C} ===")
        ts_t = time_torch(N, chi, B, reps, ham, runs)
        print(f"torch : {fmt(ts_t)}")
        ts_j = time_jax(N, chi, B, reps, ham, runs)
        print(f"jax   : {fmt(ts_j)}")
        ratio = statistics.median(ts_t) / statistics.median(ts_j)
        verdict = "JAX faster" if ratio > 1 else "torch faster"
        print(f"→ torch/jax = {ratio:.2f}x  ({verdict})")


if __name__ == "__main__":
    main()
