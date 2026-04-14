"""Tests for TREVEstimator (Qiskit BaseEstimatorV2)."""

import math
import pytest
import numpy as np

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator

from qiskit_trev.estimator import TREVEstimator, _observables_to_sparse_pauli_op
from qiskit_trev.tensor_ring.state import GateInstruction


class TestBasicEstimation:

    def test_Z_on_zero(self):
        """<0|Z|0> = 1.0."""
        qc = QuantumCircuit(1)
        op = SparsePauliOp.from_list([("Z", 1.0)])
        est = TREVEstimator(rank=1)
        result = est.run([(qc, op)]).result()
        np.testing.assert_allclose(result[0].data.evs, 1.0, atol=1e-5)

    def test_ZZ_on_bell(self):
        """<Bell|ZZ|Bell> = 1.0."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        op = SparsePauliOp.from_list([("ZZ", 1.0)])
        est = TREVEstimator(rank=4)
        result = est.run([(qc, op)]).result()
        np.testing.assert_allclose(result[0].data.evs, 1.0, atol=1e-4)

    def test_ZI_on_bell(self):
        """<Bell|ZI|Bell> = 0.0."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        op = SparsePauliOp.from_list([("ZI", 1.0)])
        est = TREVEstimator(rank=4)
        result = est.run([(qc, op)]).result()
        np.testing.assert_allclose(result[0].data.evs, 0.0, atol=1e-4)

    def test_identity(self):
        """<psi|II|psi> = 1.0."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.ry(0.7, 1)
        op = SparsePauliOp.from_list([("II", 1.0)])
        est = TREVEstimator(rank=4)
        result = est.run([(qc, op)]).result()
        np.testing.assert_allclose(result[0].data.evs, 1.0, atol=1e-5)


class TestParameterizedCircuit:

    def test_single_param(self):
        """RY(theta)|0> with Z: <Z> = cos(theta)."""
        theta = Parameter('t')
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)
        op = SparsePauliOp.from_list([("Z", 1.0)])
        est = TREVEstimator(rank=1)
        result = est.run([(qc, op, [0.5])]).result()
        np.testing.assert_allclose(result[0].data.evs, math.cos(0.5), atol=1e-4)

    def test_multiple_param_values(self):
        """Multiple parameter bindings in one PUB."""
        theta = Parameter('t')
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)
        op = SparsePauliOp.from_list([("Z", 1.0)])
        est = TREVEstimator(rank=1)
        params = [[0.0], [math.pi / 2], [math.pi]]
        result = est.run([(qc, op, params)]).result()
        evs = result[0].data.evs
        np.testing.assert_allclose(evs[0], 1.0, atol=1e-4)   # cos(0)
        np.testing.assert_allclose(evs[1], 0.0, atol=1e-4)   # cos(pi/2)
        np.testing.assert_allclose(evs[2], -1.0, atol=1e-4)  # cos(pi)


class TestMultiplePubs:

    def test_two_pubs(self):
        """Run two PUBs in one call."""
        qc1 = QuantumCircuit(1)
        qc2 = QuantumCircuit(1)
        qc2.x(0)
        op = SparsePauliOp.from_list([("Z", 1.0)])
        est = TREVEstimator(rank=1)
        result = est.run([(qc1, op), (qc2, op)]).result()
        np.testing.assert_allclose(result[0].data.evs, 1.0, atol=1e-5)
        np.testing.assert_allclose(result[1].data.evs, -1.0, atol=1e-5)


class TestMatchesStatevectorEstimator:

    def test_bell_ZZ(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        op = SparsePauliOp.from_list([("ZZ", 1.0)])

        ref = StatevectorEstimator().run([(qc, op)]).result()[0].data.evs
        trev = TREVEstimator(rank=4).run([(qc, op)]).result()[0].data.evs
        np.testing.assert_allclose(trev, ref, atol=1e-4)

    def test_multi_term(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.ry(0.8, 1)
        qc.cx(0, 1)
        op = SparsePauliOp.from_list([("ZZ", 0.5), ("ZI", -0.3), ("IZ", 0.2)])

        ref = StatevectorEstimator().run([(qc, op)]).result()[0].data.evs
        trev = TREVEstimator(rank=4).run([(qc, op)]).result()[0].data.evs
        np.testing.assert_allclose(trev, ref, atol=1e-4)

    def test_parameterized(self):
        theta = Parameter('t')
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)
        op = SparsePauliOp.from_list([("Z", 1.0)])

        ref = StatevectorEstimator().run([(qc, op, [0.7])]).result()[0].data.evs
        trev = TREVEstimator(rank=1).run([(qc, op, [0.7])]).result()[0].data.evs
        np.testing.assert_allclose(trev, ref, atol=1e-4)


class TestObservablesToSparsePauliOp:

    def test_passthrough_sparse_pauli_op(self):
        """SparsePauliOp input should be returned as-is (line 27)."""
        op = SparsePauliOp.from_list([("ZZ", 1.0)])
        result = _observables_to_sparse_pauli_op(op)
        assert result is op

    def test_dict_input(self):
        """Dict input should be converted to SparsePauliOp (line 29)."""
        result = _observables_to_sparse_pauli_op({"ZI": 0.5, "IZ": -0.3})
        assert isinstance(result, SparsePauliOp)
        pauli_list = result.to_list()
        labels = [p for p, _ in pauli_list]
        assert "ZI" in labels
        assert "IZ" in labels

    def test_unsupported_type_raises(self):
        """Unsupported type should raise TypeError (line 30)."""
        with pytest.raises(TypeError, match="Unsupported observable type"):
            _observables_to_sparse_pauli_op([("ZZ", 1.0)])


class TestNonZIHamiltonian:

    def test_x_hamiltonian(self):
        """X Hamiltonian (non-ZI) uses ev_full path (line 104)."""
        qc = QuantumCircuit(1)
        qc.h(0)
        op = SparsePauliOp.from_list([("X", 1.0)])
        est = TREVEstimator(rank=1)
        result = est.run([(qc, op)]).result()
        # <+|X|+> = 1.0
        np.testing.assert_allclose(result[0].data.evs, 1.0, atol=1e-4)

    def test_y_hamiltonian(self):
        """Y Hamiltonian (non-ZI) uses ev_full path."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.rz(math.pi / 2, 0)
        op = SparsePauliOp.from_list([("Y", 1.0)])
        ref = StatevectorEstimator().run([(qc, op)]).result()[0].data.evs
        est = TREVEstimator(rank=1)
        result = est.run([(qc, op)]).result()
        np.testing.assert_allclose(result[0].data.evs, ref, atol=1e-4)

    def test_xy_multi_term(self):
        """Multi-term XY Hamiltonian uses ev_full."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(1)
        op = SparsePauliOp.from_list([("XX", 1.0), ("YY", -0.5)])
        ref = StatevectorEstimator().run([(qc, op)]).result()[0].data.evs
        trev = TREVEstimator(rank=4).run([(qc, op)]).result()[0].data.evs
        np.testing.assert_allclose(trev, ref, atol=1e-4)


class TestZeroDimParams:

    def test_no_parameter_circuit_runs(self):
        """No-parameter circuit should work (ndim==0 branch, line 79)."""
        qc = QuantumCircuit(1)
        qc.x(0)
        op = SparsePauliOp.from_list([("Z", 1.0)])
        est = TREVEstimator(rank=1)
        # Pass without parameter values → produces 0-dim params array
        result = est.run([(qc, op)]).result()
        np.testing.assert_allclose(result[0].data.evs, -1.0, atol=1e-5)


class TestBindParams:

    def test_bind_params_fixed_gates(self):
        """_bind_params with no-param gate templates returns same gates."""
        est = TREVEstimator(rank=1)
        templates = [
            GateInstruction("H", (0,)),
            GateInstruction("X", (1,)),
        ]
        result = est._bind_params(templates, [])
        assert len(result) == 2
        assert result[0].name == "H"
        assert result[1].name == "X"

    def test_bind_params_parameterized_gates(self):
        """_bind_params substitutes parameter values into templates."""
        est = TREVEstimator(rank=1)
        templates = [
            GateInstruction("RY", (0,), params=(0.0,)),
            GateInstruction("RX", (1,), params=(0.0,)),
        ]
        result = est._bind_params(templates, [1.5, 2.5])
        assert len(result) == 2
        assert result[0].params == (1.5,)
        assert result[1].params == (2.5,)

    def test_bind_params_insufficient_values(self):
        """_bind_params falls back to original params when values run out."""
        est = TREVEstimator(rank=1)
        templates = [
            GateInstruction("RY", (0,), params=(0.7,)),
            GateInstruction("RY", (1,), params=(0.3,)),
        ]
        # Only provide enough values for first gate
        result = est._bind_params(templates, [1.5])
        assert result[0].params == (1.5,)
        # Second gate falls back to original params
        assert result[1].params == (0.3,)
