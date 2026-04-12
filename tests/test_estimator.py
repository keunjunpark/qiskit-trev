"""Tests for TREVEstimator (Qiskit BaseEstimatorV2)."""

import math
import pytest
import numpy as np

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator

from qiskit_trev.estimator import TREVEstimator


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
