"""Tests for TREVSampler (Qiskit BaseSamplerV2)."""

import pytest
import numpy as np

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.primitives import StatevectorSampler

from qiskit_trev.sampler import TREVSampler


class TestBasicSampling:

    def test_zero_state(self):
        """All-zeros circuit should give all '00' counts."""
        qc = QuantumCircuit(2)
        qc.measure_all()
        sampler = TREVSampler(rank=4)
        result = sampler.run([qc], shots=100).result()
        counts = result[0].data.meas.get_counts()
        assert counts.get("00", 0) == 100

    def test_bell_state(self):
        """Bell state should give ~50/50 between '00' and '11'."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        sampler = TREVSampler(rank=4)
        result = sampler.run([qc], shots=10000).result()
        counts = result[0].data.meas.get_counts()
        assert counts.get("00", 0) > 4000
        assert counts.get("11", 0) > 4000
        assert counts.get("01", 0) < 100
        assert counts.get("10", 0) < 100

    def test_x_gate(self):
        """X|0> should give all '1'."""
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.measure_all()
        sampler = TREVSampler(rank=1)
        result = sampler.run([qc], shots=100).result()
        counts = result[0].data.meas.get_counts()
        assert counts.get("1", 0) == 100

    def test_shot_count_respected(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.measure_all()
        sampler = TREVSampler(rank=1)
        result = sampler.run([qc], shots=500).result()
        counts = result[0].data.meas.get_counts()
        total = sum(counts.values())
        assert total == 500

    def test_returns_bit_array(self):
        """Result should have BitArray accessible via .meas."""
        qc = QuantumCircuit(2)
        qc.measure_all()
        sampler = TREVSampler(rank=4)
        result = sampler.run([qc], shots=10).result()
        data = result[0].data
        assert hasattr(data, 'meas')


class TestParameterized:

    def test_parameterized_circuit(self):
        """Parameterized circuit with binding."""
        theta = Parameter('t')
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)
        qc.measure_all()
        sampler = TREVSampler(rank=1)
        # theta=pi → should give all '1'
        import math
        result = sampler.run([(qc, [math.pi])], shots=100).result()
        counts = result[0].data.meas.get_counts()
        assert counts.get("1", 0) > 95


class TestMultiplePubs:

    def test_two_pubs(self):
        qc1 = QuantumCircuit(1)
        qc1.measure_all()
        qc2 = QuantumCircuit(1)
        qc2.x(0)
        qc2.measure_all()
        sampler = TREVSampler(rank=1)
        result = sampler.run([qc1, qc2], shots=100).result()
        c1 = result[0].data.meas.get_counts()
        c2 = result[1].data.meas.get_counts()
        assert c1.get("0", 0) == 100
        assert c2.get("1", 0) == 100


class TestMatchesStatevectorSampler:

    def test_ghz_distribution(self):
        """GHZ state: compare distributions."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure_all()

        shots = 10000
        ref = StatevectorSampler().run([qc], shots=shots).result()
        trev = TREVSampler(rank=4).run([qc], shots=shots).result()

        ref_counts = ref[0].data.meas.get_counts()
        trev_counts = trev[0].data.meas.get_counts()

        # Both should have ~50% '000' and ~50% '111'
        for key in ["000", "111"]:
            ref_frac = ref_counts.get(key, 0) / shots
            trev_frac = trev_counts.get(key, 0) / shots
            assert abs(ref_frac - trev_frac) < 0.1
