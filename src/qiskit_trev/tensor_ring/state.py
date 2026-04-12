"""Tensor ring state representation and circuit simulation.

Ported from TREV circuit.py (real_form_autograd branch).
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from . import gates as gate_fns
from .contraction import (
    apply_single_qubit_gate,
    apply_double_qubit_gate,
    apply_single_qubit_gate_batch,
    apply_double_qubit_gate_batch,
    swap_gate_matrix,
)


# Gate name → matrix function mapping
_GATE_MAP_0Q = {
    "I": gate_fns.I,
    "H": gate_fns.H,
    "X": gate_fns.X,
    "Y": gate_fns.Y,
    "Z": gate_fns.Z,
}

_GATE_MAP_1P = {
    "RX": gate_fns.RX,
    "RY": gate_fns.RY,
    "RZ": gate_fns.RZ,
}

_GATE_MAP_2Q_FIXED = {
    "CNOT": gate_fns.CNOT,
    "SWAP": gate_fns.SWAP,
}

_GATE_MAP_2Q_PARAM = {
    "ZZ": gate_fns.ZZ,
    "ZZ_SWAP": gate_fns.ZZ_SWAP,
}


@dataclass
class GateInstruction:
    """A single gate to apply to the tensor ring.

    Attributes:
        name: Gate name (e.g., "H", "RX", "CNOT").
        qubits: Tuple of qubit indices. Length 1 for single-qubit, 2 for two-qubit.
        params: Parameter values for parameterized gates. Empty for fixed gates.
    """
    name: str
    qubits: tuple[int, ...]
    params: tuple[float, ...] = ()


def _get_gate_matrix(instr: GateInstruction, device: str) -> Tensor:
    """Resolve a GateInstruction to its matrix."""
    name = instr.name
    params = instr.params

    if name in _GATE_MAP_0Q:
        return _GATE_MAP_0Q[name](device=device)
    elif name in _GATE_MAP_1P:
        return _GATE_MAP_1P[name](params[0], device=device)
    elif name == "U3":
        p = torch.tensor(params, dtype=torch.float, device=device)
        return gate_fns.U3(p, device=device)
    elif name in _GATE_MAP_2Q_FIXED:
        return _GATE_MAP_2Q_FIXED[name](device=device)
    elif name in _GATE_MAP_2Q_PARAM:
        return _GATE_MAP_2Q_PARAM[name](params[0], device=device)
    else:
        raise ValueError(f"Unknown gate: {name}")


def _is_single_qubit(instr: GateInstruction) -> bool:
    return len(instr.qubits) == 1


def _is_two_qubit(instr: GateInstruction) -> bool:
    return len(instr.qubits) == 2


def _are_adjacent(q0: int, q1: int, num_qubits: int) -> bool:
    """Check if two qubits are adjacent in the ring topology."""
    if abs(q0 - q1) == 1:
        return True
    if {q0, q1} == {0, num_qubits - 1}:
        return True
    return False


def _compile_fused_ops(gates: list[GateInstruction]):
    """Group consecutive single-qubit gates into fusible blocks.

    Yields (op_type, payload):
      - ('block1q', {qubit: [instr, ...]})
      - ('2q', instr)
    """
    ops = []
    current_block: dict[int, list[GateInstruction]] = {}

    for instr in gates:
        if _is_single_qubit(instr):
            q = instr.qubits[0]
            if q not in current_block:
                current_block[q] = []
            current_block[q].append(instr)
        else:
            if current_block:
                ops.append(('block1q', current_block))
                current_block = {}
            ops.append(('2q', instr))

    if current_block:
        ops.append(('block1q', current_block))

    return ops


class TensorRingState:
    """Tensor ring quantum state builder.

    Builds a tensor ring representation of a quantum state by applying
    a sequence of gate instructions to the initial |0...0> state.

    Attributes:
        num_qubits: Number of qubits.
        rank: Bond dimension (chi) of the tensor ring.
        device: Torch device string.
        dtype: Complex dtype for tensors.
    """

    def __init__(
        self,
        num_qubits: int,
        rank: int = 10,
        device: str = "cpu",
        dtype: torch.dtype = torch.cfloat,
    ):
        self.num_qubits = num_qubits
        self.rank = rank
        self.device = device
        self.dtype = dtype

    def build(self, gates: list[GateInstruction]) -> Tensor:
        """Build the tensor ring state by applying gates sequentially.

        Args:
            gates: List of gate instructions to apply.

        Returns:
            Tensor of shape (num_qubits, rank, rank, 2).

        Raises:
            ValueError: If a two-qubit gate targets non-adjacent qubits.
        """
        N = self.num_qubits
        rank = self.rank

        # Initialize |0...0> state
        tensor = torch.zeros(
            (N, rank, rank, 2), dtype=self.dtype, device=self.device
        )
        tensor[:, 0, 0, 0] = 1.0

        # Compile and apply
        ops = _compile_fused_ops(gates)
        for op_type, payload in ops:
            if op_type == 'block1q':
                for qubit, instrs in payload.items():
                    # Fuse consecutive single-qubit gates
                    fused = None
                    for instr in instrs:
                        mat = _get_gate_matrix(instr, self.device)
                        fused = mat if fused is None else torch.mm(mat, fused)
                    tensor[qubit] = apply_single_qubit_gate(fused, tensor[qubit])
            else:  # '2q'
                instr = payload
                q0, q1 = instr.qubits

                if not _are_adjacent(q0, q1, N):
                    raise ValueError(
                        f"Two-qubit gate {instr.name} on qubits ({q0}, {q1}) "
                        f"requires adjacent qubits in the ring topology."
                    )

                matrix = _get_gate_matrix(instr, self.device)
                self._apply_two_qubit_gate(tensor, matrix, q0, q1)

        return tensor

    def _apply_two_qubit_gate(
        self, tensor: Tensor, matrix: Tensor, q0: int, q1: int
    ) -> None:
        """Apply a two-qubit gate handling ring wrapping.

        Modifies tensor in-place.
        """
        N = self.num_qubits
        is_wrap_fwd = (q0 == N - 1 and q1 == 0)
        is_wrap_bwd = (q0 == 0 and q1 == N - 1)

        if is_wrap_fwd:
            tensor[N - 1], tensor[0] = apply_double_qubit_gate(
                matrix, tensor[N - 1], tensor[0], max_rank=self.rank
            )
        elif is_wrap_bwd:
            swapped = swap_gate_matrix(matrix)
            tensor[N - 1], tensor[0] = apply_double_qubit_gate(
                swapped, tensor[N - 1], tensor[0], max_rank=self.rank
            )
        elif q0 < q1:
            tensor[q0], tensor[q1] = apply_double_qubit_gate(
                matrix, tensor[q0], tensor[q1], max_rank=self.rank
            )
        else:
            # q0 > q1, non-wrap
            swapped = swap_gate_matrix(matrix)
            tensor[q1], tensor[q0] = apply_double_qubit_gate(
                swapped, tensor[q1], tensor[q0], max_rank=self.rank
            )

    def build_batch(
        self, gates: list[GateInstruction], params_batch: Tensor
    ) -> Tensor:
        """Build tensor ring states for a batch of parameter sets.

        Args:
            gates: Gate instruction templates (params are placeholders).
            params_batch: (B, P) tensor of parameter values per batch element.

        Returns:
            Tensor of shape (B, num_qubits, rank, rank, 2).
        """
        B = params_batch.shape[0]
        N = self.num_qubits
        rank = self.rank

        # Initialize batched |0...0> state
        single = torch.zeros(
            (N, rank, rank, 2), dtype=self.dtype, device=self.device
        )
        single[:, 0, 0, 0] = 1.0
        tensor = single.unsqueeze(0).expand(B, -1, -1, -1, -1).clone()

        ops = _compile_fused_ops(gates)
        param_idx = 0

        for op_type, payload in ops:
            if op_type == 'block1q':
                for qubit, instrs in payload.items():
                    fused = None
                    for instr in instrs:
                        mat, p_consumed = self._get_batch_gate_matrix(
                            instr, params_batch, param_idx, B
                        )
                        param_idx += p_consumed
                        fused = mat if fused is None else torch.bmm(mat, fused)
                    tensor[:, qubit] = apply_single_qubit_gate_batch(
                        fused, tensor[:, qubit]
                    )
            else:  # '2q'
                instr = payload
                q0, q1 = instr.qubits

                if not _are_adjacent(q0, q1, N):
                    raise ValueError(
                        f"Two-qubit gate {instr.name} on qubits ({q0}, {q1}) "
                        f"requires adjacent qubits in the ring topology."
                    )

                matrix, p_consumed = self._get_batch_gate_matrix_2q(
                    instr, params_batch, param_idx, B
                )
                param_idx += p_consumed
                self._apply_two_qubit_gate_batch(tensor, matrix, q0, q1)

        return tensor

    def _get_batch_gate_matrix(
        self, instr: GateInstruction, params_batch: Tensor, param_idx: int, B: int
    ) -> tuple[Tensor, int]:
        """Get batched gate matrix for a single-qubit gate.

        Returns (batch_matrix, num_params_consumed).
        """
        name = instr.name

        if name in _GATE_MAP_0Q:
            mat = _GATE_MAP_0Q[name](device=self.device)
            return mat.unsqueeze(0).expand(B, -1, -1).clone(), 0

        if name in _GATE_MAP_1P:
            theta_batch = params_batch[:, param_idx]
            mat = _GATE_MAP_1P[name](theta_batch, device=self.device)
            return mat, 1

        if name == "U3":
            p = params_batch[:, param_idx:param_idx + 3]
            mat = gate_fns.U3(p, device=self.device)
            return mat, 3

        raise ValueError(f"Unknown 1q gate: {name}")

    def _get_batch_gate_matrix_2q(
        self, instr: GateInstruction, params_batch: Tensor, param_idx: int, B: int
    ) -> tuple[Tensor, int]:
        """Get gate matrix for a two-qubit gate (same for all batch elements).

        Returns (matrix, num_params_consumed).
        """
        name = instr.name

        if name in _GATE_MAP_2Q_FIXED:
            return _GATE_MAP_2Q_FIXED[name](device=self.device), 0

        if name in _GATE_MAP_2Q_PARAM:
            theta_batch = params_batch[:, param_idx]
            mat = _GATE_MAP_2Q_PARAM[name](theta_batch, device=self.device)
            return mat, 1

        raise ValueError(f"Unknown 2q gate: {name}")

    def _apply_two_qubit_gate_batch(
        self, tensor: Tensor, matrix: Tensor, q0: int, q1: int
    ) -> None:
        """Apply a two-qubit gate to batched tensor, handling ring wrapping."""
        N = self.num_qubits
        is_wrap_fwd = (q0 == N - 1 and q1 == 0)
        is_wrap_bwd = (q0 == 0 and q1 == N - 1)

        if is_wrap_fwd:
            tensor[:, N - 1], tensor[:, 0] = apply_double_qubit_gate_batch(
                matrix, tensor[:, N - 1], tensor[:, 0], max_rank=self.rank
            )
        elif is_wrap_bwd:
            swapped = swap_gate_matrix(matrix) if matrix.ndim == 2 else torch.stack(
                [swap_gate_matrix(matrix[i]) for i in range(matrix.shape[0])]
            )
            tensor[:, N - 1], tensor[:, 0] = apply_double_qubit_gate_batch(
                swapped, tensor[:, N - 1], tensor[:, 0], max_rank=self.rank
            )
        elif q0 < q1:
            tensor[:, q0], tensor[:, q1] = apply_double_qubit_gate_batch(
                matrix, tensor[:, q0], tensor[:, q1], max_rank=self.rank
            )
        else:
            swapped = swap_gate_matrix(matrix) if matrix.ndim == 2 else torch.stack(
                [swap_gate_matrix(matrix[i]) for i in range(matrix.shape[0])]
            )
            tensor[:, q1], tensor[:, q0] = apply_double_qubit_gate_batch(
                swapped, tensor[:, q1], tensor[:, q0], max_rank=self.rank
            )
