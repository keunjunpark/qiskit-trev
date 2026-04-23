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

    ops = _compile_fused_ops(gates)

    for op_type, payload in ops:
        if op_type == "block1q":
            for qubit, instrs in payload.items():
                fused = None
                for instr in instrs:
                    mat = _batched_gate_1q(instr, params_batch, B)
                    fused = mat if fused is None else jnp.matmul(
                        mat, fused, precision=precision
                    )
                new_core = apply_single_qubit_gate_batch_jax(
                    fused, tensor[:, qubit], precision=precision
                )
                tensor = tensor.at[:, qubit].set(new_core)
        else:  # '2q'
            instr = payload
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
