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
from .measure.efficient_contraction import batched_expectation_value as batched_ev_efficient
from .measure.full_contraction import expectation_value as ev_full
from .measure.full_contraction import batched_expectation_value as batched_ev_full


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
    def evaluate_batch(
        self, params_batch: Tensor, batch_size: int | None = None
    ) -> Tensor:
        """Evaluate expectation values for a batch of parameter sets.

        Args:
            params_batch: (B, P) tensor of parameter values.
            batch_size: Chunk size for batched building. None = all at once.

        Returns:
            (B,) tensor of expectation values.
        """
        B = params_batch.shape[0]
        if batch_size is None:
            batch_size = B

        evs = torch.zeros(B, dtype=torch.float64, device=torch.device(self.device_str))

        for start in range(0, B, batch_size):
            stop = min(start + batch_size, B)
            chunk = params_batch[start:stop]

            state = TensorRingState(
                self._num_qubits, self.rank, self.device_str, self.dtype
            )
            batch_tensor = state.build_batch(self._gate_templates, chunk)

            if self._use_efficient:
                evs[start:stop] = batched_ev_efficient(batch_tensor, self._hamiltonian)
            else:
                evs[start:stop] = batched_ev_full(batch_tensor, self._hamiltonian)

        return evs

    @torch.no_grad()
    def parameter_shift_grad(
        self,
        params: Tensor,
        shift: float = math.pi / 2,
        batch_size: int | None = None,
    ) -> Tensor:
        """Compute gradient via parameter-shift rule.

        Builds all 2*P shifted parameter sets and evaluates them in one
        batched call instead of 2*P sequential forward passes.

        Args:
            params: (P,) tensor of circuit parameter values.
            shift: Shift amount (default pi/2 for standard gates).
            batch_size: Chunk size for evaluate_batch. None = all at once.

        Returns:
            (P,) tensor of gradients.
        """
        P = len(params)
        if P == 0:
            return torch.zeros(0, dtype=torch.float64)

        denom = 2 * math.sin(shift)

        # Build (2*P, P) shifted parameter matrix
        shifted = params.unsqueeze(0).expand(2 * P, -1).clone()
        idx = torch.arange(P)
        shifted[idx, idx] += shift
        shifted[P + idx, idx] -= shift

        evs = self.evaluate_batch(shifted, batch_size=batch_size)
        grad = (evs[:P] - evs[P:]) / denom

        return grad.cpu()

    @torch.no_grad()
    def parameter_shift_grad_batch(
        self,
        params_batch: Tensor,
        shift: float = math.pi / 2,
        batch_size: int | None = None,
    ) -> Tensor:
        """Compute gradient for N data points at once.

        For each sample n and each parameter i, shifts parameter i by +/-shift
        and evaluates all 2*P*N circuits in one batched call.

        Args:
            params_batch: (N, P) tensor — different param vectors per sample.
            shift: Shift amount (default pi/2).
            batch_size: Chunk size for evaluate_batch. None = all at once.

        Returns:
            (N, P) gradient tensor.
        """
        N, P = params_batch.shape
        if P == 0:
            return torch.zeros(N, 0, dtype=torch.float64)

        denom = 2 * math.sin(shift)

        # Build (2*P*N, P) mega-batch: for each param i, stack plus/minus shifted
        all_shifted = []
        for i in range(P):
            plus = params_batch.clone()
            plus[:, i] += shift
            minus = params_batch.clone()
            minus[:, i] -= shift
            all_shifted.append(plus)
            all_shifted.append(minus)

        mega = torch.cat(all_shifted, dim=0)  # (2*P*N, P)
        evs = self.evaluate_batch(mega, batch_size=batch_size)  # (2*P*N,)

        # Reshape: (P, 2, N) — for each param, plus then minus, across N samples
        evs = evs.view(P, 2, N)
        grad = (evs[:, 0, :] - evs[:, 1, :]) / denom  # (P, N)

        return grad.T.cpu()  # (N, P)
