"""TensorRingModel: PyTorch nn.Module wrapping a quantum circuit + observable.

Gradients are computed via parameter-shift rule (no autograd).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from .converter import circuit_to_gate_instructions, sparse_pauli_op_to_hamiltonian
from .tensor_ring.state import TensorRingState, GateInstruction
from .hamiltonian import Hamiltonian
from .measure.efficient_contraction import expectation_value as ev_efficient
from .measure.full_contraction import expectation_value as ev_full


class TensorRingModel(torch.nn.Module):
    """Tensor ring quantum circuit model as a PyTorch nn.Module.

    Wraps a parameterized Qiskit QuantumCircuit and an observable
    (SparsePauliOp) into a callable model. The forward pass computes
    the expectation value <psi(params)|H|psi(params)>.

    Gradients are computed via parameter-shift rule, not autograd.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        observable: SparsePauliOp,
        *,
        rank: int = 10,
        device: str = "cpu",
        dtype: torch.dtype = torch.cfloat,
    ):
        super().__init__()
        self.rank = rank
        self.device_str = device
        self.dtype = dtype

        # Convert circuit to gate instructions (with placeholder params)
        self._gate_templates, self._num_params = circuit_to_gate_instructions(circuit)
        self._num_qubits = circuit.num_qubits

        # Convert observable
        self._hamiltonian = sparse_pauli_op_to_hamiltonian(observable)

        # Choose measurement method based on Hamiltonian type
        self._use_efficient = self._hamiltonian.has_only_zi

    def _build_gates(self, params: Tensor) -> list[GateInstruction]:
        """Build gate list with parameter values substituted."""
        gates = []
        param_idx = 0
        for tmpl in self._gate_templates:
            if tmpl.params:
                n_p = len(tmpl.params)
                if param_idx + n_p <= len(params):
                    new_params = tuple(params[param_idx + i].item() for i in range(n_p))
                    param_idx += n_p
                else:
                    new_params = tmpl.params
                gates.append(GateInstruction(tmpl.name, tmpl.qubits, new_params))
            else:
                gates.append(tmpl)
        return gates

    def forward(self, params: Tensor) -> Tensor:
        """Compute <psi(params)|H|psi(params)>.

        Args:
            params: (P,) tensor of circuit parameter values.

        Returns:
            Scalar tensor (expectation value).
        """
        gates = self._build_gates(params)
        state = TensorRingState(
            self._num_qubits, self.rank, self.device_str, self.dtype
        )
        tensor = state.build(gates)

        if self._use_efficient:
            ev = ev_efficient(tensor, self._hamiltonian)
        else:
            ev = ev_full(tensor, self._hamiltonian)

        return torch.tensor(ev, dtype=torch.float64)

    @torch.no_grad()
    def parameter_shift_grad(
        self, params: Tensor, shift: float = math.pi / 2
    ) -> Tensor:
        """Compute gradient via parameter-shift rule.

        For each parameter i:
            grad[i] = (E(theta + shift*e_i) - E(theta - shift*e_i)) / (2*sin(shift))

        Args:
            params: (P,) tensor of circuit parameter values.
            shift: Shift amount (default pi/2 for standard gates).

        Returns:
            (P,) tensor of gradients.
        """
        P = len(params)
        if P == 0:
            return torch.zeros(0, dtype=torch.float64)

        grad = torch.zeros(P, dtype=torch.float64)
        denom = 2 * math.sin(shift)

        for i in range(P):
            params_plus = params.clone()
            params_plus[i] += shift
            params_minus = params.clone()
            params_minus[i] -= shift

            ev_plus = self.forward(params_plus).item()
            ev_minus = self.forward(params_minus).item()
            grad[i] = (ev_plus - ev_minus) / denom

        return grad
