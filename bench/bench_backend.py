"""Backend benchmark: torch vs JAX (eager and jit) on batched_expectation_value.

Primary go/no-go gate for plan/14 Step 2.

Usage:
    .venv/bin/python bench/bench_backend.py
"""

from __future__ import annotations

import statistics
import time

import numpy as np
import torch

import jax
import jax.numpy as jnp

from qiskit_trev.backend import JAX_BACKEND, JAX_BACKEND_HIGHEST
from qiskit_trev.hamiltonian import Hamiltonian
from qiskit_trev.measure.efficient_contraction import (
    _batched_expectation_kernel,
    batched_expectation_value,
)


def build_workload(chi: int, N: int, C: int, B: int, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    batch = torch.randn(B, N, chi, chi, 2, dtype=torch.cfloat, generator=gen) / (chi ** 0.5)
    paulis = []
    for i in range(C):
        s = "".join("Z" if (i >> j) & 1 else "I" for j in range(N))
        paulis.append((s, 0.5 + 0.01 * i))
    ham = Hamiltonian.from_pauli_list(paulis)
    return batch, ham


def time_torch(batch: torch.Tensor, ham: Hamiltonian, runs: int) -> list[float]:
    # warm-up
    for _ in range(3):
        batched_expectation_value(batch, ham)
        if batch.is_cuda:
            torch.cuda.synchronize()

    ts = []
    for _ in range(runs):
        if batch.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        batched_expectation_value(batch, ham)
        if batch.is_cuda:
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return ts


def _jax_inputs(batch_t: torch.Tensor, ham: Hamiltonian, backend, device):
    batch_j = jax.device_put(jnp.asarray(batch_t.cpu().numpy()), device)
    paulis = jax.device_put(
        jnp.asarray(ham.get_bool_pauli_tensor().numpy()), device
    )
    coeffs = jax.device_put(
        jnp.asarray(np.asarray(ham.coefficients, dtype=np.complex64)), device
    )
    Z_op = jax.device_put(
        jnp.asarray([[1, 0], [0, -1]], dtype=jnp.complex64), device
    )
    I_op = jax.device_put(jnp.eye(2, dtype=jnp.complex64), device)
    total_init = jax.device_put(
        jnp.zeros(batch_t.shape[0], dtype=jnp.complex64), device
    )
    return batch_j, paulis, coeffs, Z_op, I_op, total_init


def time_jax_jit(batch_t: torch.Tensor, ham: Hamiltonian, backend, runs: int):
    device = jax.devices()[0]
    batch_j, paulis, coeffs, Z_op, I_op, total_init = _jax_inputs(
        batch_t, ham, backend, device
    )

    jit_kernel = jax.jit(_batched_expectation_kernel, static_argnums=(0, 7))

    # compile + warm-up
    t0 = time.perf_counter()
    jit_kernel(
        backend, batch_j, paulis, coeffs, Z_op, I_op, total_init, None
    ).block_until_ready()
    compile_s = time.perf_counter() - t0
    for _ in range(2):
        jit_kernel(
            backend, batch_j, paulis, coeffs, Z_op, I_op, total_init, None
        ).block_until_ready()

    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        jit_kernel(
            backend, batch_j, paulis, coeffs, Z_op, I_op, total_init, None
        ).block_until_ready()
        ts.append(time.perf_counter() - t0)
    return ts, compile_s


def time_jax_eager(batch_t: torch.Tensor, ham: Hamiltonian, runs: int):
    device = jax.devices()[0]
    batch_j = jax.device_put(jnp.asarray(batch_t.cpu().numpy()), device)

    # warm-up
    for _ in range(3):
        batched_expectation_value(batch_j, ham).block_until_ready()

    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        batched_expectation_value(batch_j, ham).block_until_ready()
        ts.append(time.perf_counter() - t0)
    return ts


def fmt(ts: list[float]) -> str:
    return (
        f"median={statistics.median(ts)*1e3:8.2f} ms  "
        f"min={min(ts)*1e3:8.2f} ms  "
        f"stdev={statistics.stdev(ts)*1e3:6.2f} ms"
    )


def main():
    torch.set_num_threads(1)
    # On non-CUDA hosts (CPU, TPU), torch runs on CPU and dominates wall time.
    # Cut runs and skip the two extra-JAX paths kept for local precision study.
    has_cuda = torch.cuda.is_available()
    runs = 20 if has_cuda else 5

    for chi, N, C, B in [(12, 16, 100, 32), (8, 12, 50, 16)]:
        print(f"\n=== χ={chi} N={N} C={C} B={B} ===")
        batch, ham = build_workload(chi, N, C, B)
        device_t = torch.device("cuda" if has_cuda else "cpu")
        batch = batch.to(device_t)

        ts_torch = time_torch(batch, ham, runs)
        print(f"torch           : {fmt(ts_torch)}")

        ts_jax_jit, compile_s = time_jax_jit(batch, ham, JAX_BACKEND, runs)
        print(f"jax jit   DEF   : {fmt(ts_jax_jit)}  "
              f"(first-call compile={compile_s*1e3:.1f} ms)")

        ratio_def = statistics.median(ts_torch) / statistics.median(ts_jax_jit)
        print(f"→ torch/jax-jit (DEFAULT)  = {ratio_def:.2f}x")

        if has_cuda:
            ts_jax_eager = time_jax_eager(batch, ham, runs)
            print(f"jax eager DEF   : {fmt(ts_jax_eager)}")

            ts_jax_high, compile_s_h = time_jax_jit(
                batch, ham, JAX_BACKEND_HIGHEST, runs
            )
            print(f"jax jit   HIGH  : {fmt(ts_jax_high)}  "
                  f"(first-call compile={compile_s_h*1e3:.1f} ms)")
            ratio_high = statistics.median(ts_torch) / statistics.median(ts_jax_high)
            print(f"→ torch/jax-jit (HIGHEST)  = {ratio_high:.2f}x")


if __name__ == "__main__":
    main()
