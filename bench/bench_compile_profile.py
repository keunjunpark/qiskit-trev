"""Compile-time profile — where does JAX compile time actually go?

JAX exposes the compile pipeline in stages:

    python/jaxpr ── trace+lower ──▶ HLO (IR) ── XLA compile ──▶ executable

This bench times the two stages separately and reports HLO size so we
can see:
1. Is compile dominated by lowering (Python side) or XLA optimisation?
2. Does HLO size grow linearly, quadratically, or worse with χ and gate
   count? Sets the scaling law for plan/15 (Track B) reopen gate.
3. Does the second invocation of a compiled callable hit the jit cache?

Run on GPU. Output is a table of (χ, reps) vs (lower_s, compile_s,
hlo_kbytes, first_run_ms, steady_run_ms).

Usage:
    python bench/bench_compile_profile.py
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

from qiskit_trev.backend import JAX_BACKEND
from qiskit_trev.measure.efficient_contraction import _batched_expectation_kernel
from qiskit_trev.model import TensorRingModel
from qiskit_trev.tensor_ring._state_jax import build_batch_jax


def build(N: int, reps: int, rank: int, C: int, device: str, ansatz: str = "ry_cnot"):
    """Build a model for the compile-profile bench.

    ``ansatz`` selects the entangling layer:
      - ``"ry_cnot"``: RY layer + CNOT ring (hardware-efficient VQE).
      - ``"ry_zz"``: RY layer + RZZ ring (QAOA / Hamiltonian-variational).
        Qiskit's ``qc.rzz`` maps to the internal ``ZZ`` gate, which
        exercises the parameterised-2q-run fori_loop fast path.
    """
    qc = QuantumCircuit(N)
    P = 0
    for _ in range(reps):
        for q in range(N):
            qc.ry(0.0, q)
            P += 1
        if ansatz == "ry_cnot":
            for q in range(N):
                qc.cx(q, (q + 1) % N)
        elif ansatz == "ry_zz":
            for q in range(N):
                qc.rzz(0.0, q, (q + 1) % N)
                P += 1
        else:
            raise ValueError(f"unknown ansatz {ansatz!r}")
    rng = np.random.RandomState(0)
    terms = []
    for _ in range(C):
        s = "".join(rng.choice(["I", "Z"]) for _ in range(N))
        if "Z" not in s:
            s = "Z" + s[1:]
        terms.append((s, float(rng.rand() - 0.5)))
    obs = SparsePauliOp.from_list(terms)
    return TensorRingModel(qc, obs, rank=rank, device=device), P


def _hlo_kb(lowered) -> float:
    """Approximate HLO size for the lowered program."""
    txt = lowered.as_text()
    return len(txt) / 1024.0


def _hlo_instr_count(lowered) -> int:
    txt = lowered.as_text()
    return sum(1 for ln in txt.splitlines() if ln.strip().startswith("%"))


def profile_grad_kernel(
    N: int, reps: int, rank: int, C: int, device: str, ansatz: str = "ry_cnot"
):
    """Build the same jit'd gradient kernel BatchParameterShiftGradient uses
    and profile its compile stages.

    Returns dict of stage timings and HLO size.
    """
    model, P = build(N, reps, rank, C, device, ansatz=ansatz)

    ham = model._hamiltonian
    paulis = jnp.asarray(ham.get_bool_pauli_tensor().numpy())
    coeffs = jnp.asarray(np.asarray(ham.coefficients, dtype=np.complex64))
    Z_op = jnp.asarray([[1, 0], [0, -1]], dtype=jnp.complex64)
    I_op = jnp.eye(2, dtype=jnp.complex64)

    shift = math.pi / 2
    denom = 2.0 * math.sin(shift)

    num_qubits = model._num_qubits
    r = model.rank
    gates = model._gate_templates

    def grad_fn(params_j):
        eye = jnp.eye(P, dtype=params_j.dtype)
        all_shifted = jnp.concatenate(
            [params_j[None] + shift * eye, params_j[None] - shift * eye], axis=0
        )
        state = build_batch_jax(num_qubits, r, gates, all_shifted)
        total_init = jnp.zeros(2 * P, dtype=jnp.complex64)
        evs = _batched_expectation_kernel(
            JAX_BACKEND, state, paulis, coeffs, Z_op, I_op, total_init, None
        )
        return (evs[:P] - evs[P:]) / denom

    params = jnp.asarray(
        (np.random.RandomState(0).rand(P) * 6.28).astype(np.float32)
    )

    # Stage 1 — trace + lower (Python → HLO)
    t0 = time.perf_counter()
    lowered = jax.jit(grad_fn).lower(params)
    t_lower = time.perf_counter() - t0

    hlo_kb = _hlo_kb(lowered)
    hlo_instr = _hlo_instr_count(lowered)

    # Stage 2 — XLA compile (HLO → executable)
    t0 = time.perf_counter()
    compiled = lowered.compile()
    t_compile = time.perf_counter() - t0

    # Stage 3 — first execution after compile
    t0 = time.perf_counter()
    compiled(params).block_until_ready()
    t_first = time.perf_counter() - t0

    # Stage 4 — steady-state
    for _ in range(2):  # extra warm-up
        compiled(params).block_until_ready()
    ts = []
    for _ in range(10):
        t0 = time.perf_counter()
        compiled(params).block_until_ready()
        ts.append(time.perf_counter() - t0)
    t_steady = statistics.median(ts)

    return {
        "N": N,
        "reps": reps,
        "rank": rank,
        "C": C,
        "P": P,
        "lower_s": t_lower,
        "compile_s": t_compile,
        "hlo_kb": hlo_kb,
        "hlo_instructions": hlo_instr,
        "first_run_ms": t_first * 1e3,
        "steady_run_ms": t_steady * 1e3,
    }


def main():
    torch.set_num_threads(1)
    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"

    # Three sweeps to isolate what drives compile:
    #   chi_sweep   — varies bond dim (the Track-B research knob)
    #   depth_sweep — varies circuit reps (conflates depth + parameter count)
    #   N_sweep     — varies qubit count at fixed reps=1 (isolates "more qubits"
    #                 from "deeper circuit")
    chi_sweep = [(6, 1, chi, 5) for chi in [6, 8, 12, 16, 24, 32]]
    depth_sweep = [(6, r, 8, 5) for r in [1, 2, 3]]
    N_sweep = [(N, 1, 8, 5) for N in [4, 6, 8, 10, 12]]
    # QAOA / HVA-style: RY + RZZ ring. Same parameters as N_sweep so
    # the comparison is apples-to-apples with the CNOT-ring numbers
    # printed above.
    zz_N_sweep = [(N, 1, 8, 5) for N in [4, 6, 8, 10, 12]]
    zz_depth_sweep = [(6, r, 8, 5) for r in [1, 2, 3]]

    print(f"device = {device}")
    print(
        f"{'N':>3} {'reps':>4} {'rank':>5} {'C':>3} {'P':>4} "
        f"{'lower(s)':>9} {'compile(s)':>11} {'hlo(kB)':>9} "
        f"{'instr':>7} {'first(ms)':>10} {'steady(ms)':>11}"
    )

    def row(r):
        print(
            f"{r['N']:>3} {r['reps']:>4} {r['rank']:>5} {r['C']:>3} {r['P']:>4} "
            f"{r['lower_s']:>9.2f} {r['compile_s']:>11.2f} {r['hlo_kb']:>9.1f} "
            f"{r['hlo_instructions']:>7} {r['first_run_ms']:>10.2f} "
            f"{r['steady_run_ms']:>11.3f}"
        )

    print(f"\n--- χ sweep (N=6, reps=1, C=5) ---")
    for N, reps, rank, C in chi_sweep:
        try:
            row(profile_grad_kernel(N, reps, rank, C, device))
        except Exception as e:
            print(f"skip rank={rank}: {type(e).__name__}: {str(e)[:160]}")

    print(f"\n--- depth sweep (N=6, rank=8, C=5) ---")
    for N, reps, rank, C in depth_sweep:
        try:
            row(profile_grad_kernel(N, reps, rank, C, device))
        except Exception as e:
            print(f"skip reps={reps}: {type(e).__name__}: {str(e)[:160]}")

    print(f"\n--- N sweep (reps=1, rank=8, C=5) ---")
    for N, reps, rank, C in N_sweep:
        try:
            row(profile_grad_kernel(N, reps, rank, C, device))
        except Exception as e:
            print(f"skip N={N}: {type(e).__name__}: {str(e)[:160]}")

    print(f"\n--- ZZ-ring (RY+RZZ) N sweep (reps=1, rank=8, C=5) ---")
    for N, reps, rank, C in zz_N_sweep:
        try:
            row(profile_grad_kernel(N, reps, rank, C, device, ansatz="ry_zz"))
        except Exception as e:
            print(f"skip N={N}: {type(e).__name__}: {str(e)[:160]}")

    print(f"\n--- ZZ-ring (RY+RZZ) depth sweep (N=6, rank=8, C=5) ---")
    for N, reps, rank, C in zz_depth_sweep:
        try:
            row(profile_grad_kernel(N, reps, rank, C, device, ansatz="ry_zz"))
        except Exception as e:
            print(f"skip reps={reps}: {type(e).__name__}: {str(e)[:160]}")


if __name__ == "__main__":
    main()
