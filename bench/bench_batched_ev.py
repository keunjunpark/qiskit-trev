"""Microbenchmark for batched_expectation_value — Step 1 refactor overhead check.

Usage:
    .venv/bin/python bench/bench_batched_ev.py
"""

from __future__ import annotations

import statistics
import time

import torch

from qiskit_trev.hamiltonian import Hamiltonian
from qiskit_trev.measure.efficient_contraction import batched_expectation_value


def build_workload(chi: int, N: int, C: int, B: int, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    batch = torch.randn(B, N, chi, chi, 2, dtype=torch.cfloat, generator=gen)
    paulis = []
    for i in range(C):
        s = "".join("Z" if (i >> j) & 1 else "I" for j in range(N))
        paulis.append((s, 0.5 + 0.01 * i))
    ham = Hamiltonian.from_pauli_list(paulis)
    return batch, ham


def time_once(batch, ham) -> float:
    if batch.is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    batched_expectation_value(batch, ham)
    if batch.is_cuda:
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def main():
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chi, N, C, B = 12, 16, 100, 32
    batch, ham = build_workload(chi, N, C, B)
    batch = batch.to(device)

    # warm-up
    for _ in range(3):
        time_once(batch, ham)

    runs = 20
    ts = [time_once(batch, ham) for _ in range(runs)]
    print(f"device={device} chi={chi} N={N} C={C} B={B}")
    print(f"median={statistics.median(ts) * 1e3:.3f} ms  "
          f"mean={statistics.mean(ts) * 1e3:.3f} ms  "
          f"stdev={statistics.stdev(ts) * 1e3:.3f} ms  "
          f"min={min(ts) * 1e3:.3f} ms")


if __name__ == "__main__":
    main()
