"""JAX implementation of :meth:`TensorRingState.build_batch`.

Parallel to the torch path in :mod:`state`. Uses the JAX gate templates and
contraction helpers. Returns a ``jax.Array`` of shape
``(B, num_qubits, rank, rank, 2)`` that can be fed straight into
:func:`batched_expectation_value` — which dispatches to the JAX backend on
``jax.Array`` inputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from . import _gates_jax as gates_j
from ._contraction_jax import (
    apply_double_qubit_gate_batch_jax,
    apply_single_qubit_gate_batch_jax,
    swap_gate_matrix_jax,
)

if TYPE_CHECKING:
    from .state import GateInstruction


_GATE_MAP_0Q = {
    "I": gates_j.I,
    "H": gates_j.H,
    "X": gates_j.X,
    "Y": gates_j.Y,
    "Z": gates_j.Z,
}

_GATE_MAP_1P = {
    "RX": gates_j.RX,
    "RY": gates_j.RY,
    "RZ": gates_j.RZ,
}

_GATE_MAP_2Q_FIXED = {
    "CNOT": gates_j.CNOT,
    "SWAP": gates_j.SWAP,
}

_GATE_MAP_2Q_PARAM = {
    "ZZ": gates_j.ZZ,
    "ZZ_SWAP": gates_j.ZZ_SWAP,
}


def _is_single_qubit(instr: "GateInstruction") -> bool:
    return len(instr.qubits) == 1


def _are_adjacent(q0: int, q1: int, num_qubits: int) -> bool:
    if abs(q0 - q1) == 1:
        return True
    if {q0, q1} == {0, num_qubits - 1}:
        return True
    return False


def _compile_fused_ops(gates: list["GateInstruction"]):
    ops: list = []
    current_block: dict[int, list] = {}
    for instr in gates:
        if _is_single_qubit(instr):
            q = instr.qubits[0]
            current_block.setdefault(q, []).append(instr)
        else:
            if current_block:
                ops.append(("block1q", current_block))
                current_block = {}
            ops.append(("2q", instr))
    if current_block:
        ops.append(("block1q", current_block))
    return ops


def _coalesce_2q_runs(ops: list, N: int) -> list:
    """Coalesce consecutive fixed-matrix, non-wrap, q0<q1 2q gates of same name.

    These runs (e.g. a CNOT chain on adjacent pairs before the ring-wrap)
    can be applied via jax.lax.fori_loop with one compiled body instead of
    one unrolled HLO subgraph per gate. Parameterised gates and ring-wrap
    gates fall through to single-gate application.
    """
    out: list = []
    i = 0
    while i < len(ops):
        op = ops[i]
        if op[0] != "2q":
            out.append(op)
            i += 1
            continue
        instr = op[1]
        if instr.params or instr.name not in ("CNOT", "SWAP"):
            out.append(op)
            i += 1
            continue
        q0, q1 = instr.qubits
        # Non-wrap, q0 < q1, adjacent only. Ring-wrap and reversed pairs
        # keep the single-gate path (cheap, rare).
        if q0 >= q1 or (q0 == 0 and q1 == N - 1) or q1 - q0 != 1:
            out.append(op)
            i += 1
            continue

        run_pairs = [(q0, q1)]
        j = i + 1
        while j < len(ops) and ops[j][0] == "2q":
            nxt = ops[j][1]
            if nxt.name != instr.name or nxt.params:
                break
            nq0, nq1 = nxt.qubits
            if nq0 >= nq1 or (nq0 == 0 and nq1 == N - 1) or nq1 - nq0 != 1:
                break
            run_pairs.append((nq0, nq1))
            j += 1

        if len(run_pairs) >= 2:
            out.append(("2q_run_fixed", instr.name, run_pairs))
            i = j
        else:
            out.append(op)
            i += 1
    return out


def _batched_gate_1q(
    instr: "GateInstruction", params_batch: jnp.ndarray, B: int
) -> jnp.ndarray:
    name = instr.name

    if name in _GATE_MAP_0Q:
        mat = _GATE_MAP_0Q[name]()
        return jnp.broadcast_to(mat, (B,) + mat.shape)

    if name in _GATE_MAP_1P:
        theta_batch = params_batch[:, instr.param_indices[0]]
        return _GATE_MAP_1P[name](theta_batch)

    if name == "U3":
        idx = jnp.asarray(list(instr.param_indices))
        p = params_batch[:, idx]
        return gates_j.U3(p)

    raise ValueError(f"Unknown 1q gate: {name}")


def _batched_gate_2q(
    instr: "GateInstruction", params_batch: jnp.ndarray, B: int
) -> jnp.ndarray:
    name = instr.name

    if name in _GATE_MAP_2Q_FIXED:
        return _GATE_MAP_2Q_FIXED[name]()

    if name in _GATE_MAP_2Q_PARAM:
        theta_batch = params_batch[:, instr.param_indices[0]]
        return _GATE_MAP_2Q_PARAM[name](theta_batch)

    raise ValueError(f"Unknown 2q gate: {name}")


def _apply_two_qubit_gate(
    tensor: jnp.ndarray,
    matrix: jnp.ndarray,
    q0: int,
    q1: int,
    N: int,
    rank: int,
    *,
    precision: jax.lax.Precision,
) -> jnp.ndarray:
    is_wrap_fwd = (q0 == N - 1 and q1 == 0)
    is_wrap_bwd = (q0 == 0 and q1 == N - 1)

    if is_wrap_fwd:
        new_a, new_b = apply_double_qubit_gate_batch_jax(
            matrix, tensor[:, N - 1], tensor[:, 0],
            max_rank=rank, precision=precision,
        )
        tensor = tensor.at[:, N - 1].set(new_a).at[:, 0].set(new_b)
    elif is_wrap_bwd:
        swapped = swap_gate_matrix_jax(matrix)
        new_a, new_b = apply_double_qubit_gate_batch_jax(
            swapped, tensor[:, N - 1], tensor[:, 0],
            max_rank=rank, precision=precision,
        )
        tensor = tensor.at[:, N - 1].set(new_a).at[:, 0].set(new_b)
    elif q0 < q1:
        new_a, new_b = apply_double_qubit_gate_batch_jax(
            matrix, tensor[:, q0], tensor[:, q1],
            max_rank=rank, precision=precision,
        )
        tensor = tensor.at[:, q0].set(new_a).at[:, q1].set(new_b)
    else:
        swapped = swap_gate_matrix_jax(matrix)
        new_a, new_b = apply_double_qubit_gate_batch_jax(
            swapped, tensor[:, q1], tensor[:, q0],
            max_rank=rank, precision=precision,
        )
        tensor = tensor.at[:, q1].set(new_a).at[:, q0].set(new_b)

    return tensor


def build_batch_jax(
    num_qubits: int,
    rank: int,
    gates: list["GateInstruction"],
    params_batch: jnp.ndarray,
    *,
    precision: jax.lax.Precision = jax.lax.Precision.DEFAULT,
) -> jnp.ndarray:
    """JAX equivalent of :meth:`TensorRingState.build_batch`.

    Mirrors the torch implementation; numerical agreement requires
    ``Precision.HIGHEST`` to overcome tf32-like matmul on GPU.

    Returns:
        ``(B, num_qubits, rank, rank, 2)`` ``jax.Array``.
    """
    B = params_batch.shape[0]
    N = num_qubits

    # Auto-assign param_indices where missing (same convention as torch path).
    param_idx = 0
    for instr in gates:
        if instr.params and not instr.param_indices:
            instr.param_indices = tuple(
                range(param_idx, param_idx + len(instr.params))
            )
        param_idx += len(instr.params)

    # |0...0> initial state, broadcast to batch.
    single = jnp.zeros((N, rank, rank, 2), dtype=jnp.complex64)
    single = single.at[:, 0, 0, 0].set(1.0)
    tensor = jnp.broadcast_to(single[None], (B, N, rank, rank, 2))

    ops = _coalesce_2q_runs(_compile_fused_ops(gates), N)

    for item in ops:
        op_type = item[0]
        if op_type == "block1q":
            payload = item[1]
            tensor = _apply_block1q_parallel(
                tensor, payload, params_batch, B, N, precision=precision
            )
        elif op_type == "2q_run_fixed":
            _, gate_name, run_pairs = item
            fixed_mat = (
                _GATE_MAP_2Q_FIXED[gate_name]()
            )  # (4, 4) constant
            tensor = _apply_2q_run_fori(
                tensor, fixed_mat, run_pairs, rank, precision=precision
            )
        else:  # '2q' single gate
            instr = item[1]
            q0, q1 = instr.qubits
            if not _are_adjacent(q0, q1, N):
                raise ValueError(
                    f"Two-qubit gate {instr.name} on qubits ({q0}, {q1}) "
                    f"requires adjacent qubits in the ring topology."
                )
            matrix = _batched_gate_2q(instr, params_batch, B)
            tensor = _apply_two_qubit_gate(
                tensor, matrix, q0, q1, N, rank, precision=precision,
            )

    return tensor


def _apply_2q_run_fori(
    tensor: jnp.ndarray,
    gate_matrix: jnp.ndarray,
    pairs: list[tuple[int, int]],
    rank: int,
    *,
    precision: jax.lax.Precision,
) -> jnp.ndarray:
    """Apply a run of fixed-matrix, non-wrap 2q gates via ``jax.lax.fori_loop``.

    All pairs are pre-filtered to satisfy ``q1 = q0 + 1`` (adjacent, no
    wrap) so the body is branch-free. Each iteration applies one gate
    with its own SVD truncation — but the SVD HLO is compiled once and
    reused across iterations instead of being unrolled per gate.
    """
    pair_arr = jnp.asarray(pairs, dtype=jnp.int32)
    num_gates = pair_arr.shape[0]

    def body(i, tensor):
        q0 = pair_arr[i, 0]
        q1 = pair_arr[i, 1]
        core_a = jnp.take(tensor, q0, axis=1)
        core_b = jnp.take(tensor, q1, axis=1)
        new_a, new_b = apply_double_qubit_gate_batch_jax(
            gate_matrix, core_a, core_b, max_rank=rank, precision=precision
        )
        tensor = jax.lax.dynamic_update_index_in_dim(
            tensor, new_a, q0, axis=1
        )
        tensor = jax.lax.dynamic_update_index_in_dim(
            tensor, new_b, q1, axis=1
        )
        return tensor

    return jax.lax.fori_loop(0, num_gates, body, tensor)


def _apply_block1q_parallel(
    tensor: jnp.ndarray,
    payload: dict,
    params_batch: jnp.ndarray,
    B: int,
    N: int,
    *,
    precision: jax.lax.Precision,
) -> jnp.ndarray:
    """Apply a block of single-qubit gates in parallel across qubits.

    The original implementation looped over qubits and updated the tensor
    slot-by-slot via ``tensor.at[:, qubit].set(...)`` — N separate einsum
    + scatter operations, which makes HLO grow linearly in N per block.

    1q gates on different qubits commute, so we can:
      1. Build per-qubit fused matrices (identity for qubits the block
         doesn't touch).
      2. Stack into ``(B, N, 2, 2)``.
      3. Apply to the full tensor with ONE einsum.

    This collapses N gate applications to 1 einsum, shrinking HLO and
    also speeding runtime (single fused op vs. N dispatched ops).
    """
    # Build (B, N, 2, 2) stacked matrices. Qubits outside the block get I.
    eye_b = jnp.broadcast_to(
        jnp.eye(2, dtype=jnp.complex64)[None], (B, 2, 2)
    )
    fused_per_qubit: list = [eye_b] * N
    for qubit, instrs in payload.items():
        fused = None
        for instr in instrs:
            mat = _batched_gate_1q(instr, params_batch, B)
            fused = mat if fused is None else jnp.matmul(
                mat, fused, precision=precision
            )
        fused_per_qubit[qubit] = fused
    all_mats = jnp.stack(fused_per_qubit, axis=1)  # (B, N, 2, 2)

    # Apply per-site: out[b,n,k,l,i] = sum_j mats[b,n,i,j] * tensor[b,n,k,l,j]
    return jnp.einsum(
        "bnij,bnklj->bnkli", all_mats, tensor, precision=precision
    )
