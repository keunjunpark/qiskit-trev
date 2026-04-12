"""Tests for tensor ring gate matrix functions."""

import math
import pytest
import torch

from qiskit_trev.tensor_ring.gates import (
    I, H, X, Y, Z, RX, RY, RZ, U3, CNOT, SWAP, ZZ, ZZ_SWAP,
)


def _is_unitary(mat, tol=1e-5):
    """Check if a matrix is unitary: U @ U^dagger = I."""
    n = mat.shape[0]
    identity = torch.eye(n, dtype=mat.dtype, device=mat.device)
    product = mat @ mat.conj().T
    return torch.allclose(product, identity, atol=tol)


# --- Fixed gates: unitarity ---

@pytest.mark.parametrize("gate_fn", [I, H, X, Y, Z])
def test_fixed_gates_are_unitary(gate_fn):
    mat = gate_fn()
    assert _is_unitary(mat)


@pytest.mark.parametrize("gate_fn", [CNOT, SWAP])
def test_two_qubit_fixed_gates_are_unitary(gate_fn):
    mat = gate_fn()
    assert _is_unitary(mat)


# --- Fixed gates: known values ---

def test_H_known_values():
    expected = (1 / math.sqrt(2)) * torch.tensor([[1, 1], [1, -1]], dtype=torch.cfloat)
    assert torch.allclose(H(), expected, atol=1e-6)


def test_X_known_values():
    expected = torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat)
    assert torch.allclose(X(), expected)


def test_Y_known_values():
    expected = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.cfloat)
    assert torch.allclose(Y(), expected)


def test_Z_known_values():
    expected = torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat)
    assert torch.allclose(Z(), expected)


def test_CNOT_known_matrix():
    expected = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ], dtype=torch.cfloat)
    assert torch.allclose(CNOT(), expected)


def test_SWAP_known_matrix():
    expected = torch.tensor([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=torch.cfloat)
    assert torch.allclose(SWAP(), expected)


# --- Parameterized gates at theta=0 → identity ---

@pytest.mark.parametrize("gate_fn", [RX, RY, RZ])
def test_parameterized_gates_at_zero_are_identity(gate_fn):
    mat = gate_fn(0.0)
    expected = torch.eye(2, dtype=torch.cfloat)
    assert torch.allclose(mat, expected, atol=1e-6)


# --- Parameterized gates: known formulas ---

def test_RX_at_pi():
    mat = RX(math.pi)
    expected = torch.tensor([[0, -1j], [-1j, 0]], dtype=torch.cfloat)
    assert torch.allclose(mat, expected, atol=1e-5)


def test_RY_at_pi():
    mat = RY(math.pi)
    expected = torch.tensor([[0, -1], [1, 0]], dtype=torch.cfloat)
    assert torch.allclose(mat, expected, atol=1e-5)


def test_RZ_at_pi():
    mat = RZ(math.pi)
    expected = torch.tensor([[-1j, 0], [0, 1j]], dtype=torch.cfloat)
    assert torch.allclose(mat, expected, atol=1e-5)


# --- Parameterized gates: unitarity ---

@pytest.mark.parametrize("gate_fn", [RX, RY, RZ])
@pytest.mark.parametrize("theta", [0, math.pi / 4, math.pi / 2, math.pi, 3 * math.pi / 2])
def test_parameterized_gates_are_unitary(gate_fn, theta):
    mat = gate_fn(theta)
    assert _is_unitary(mat)


# --- U3 ---

def test_U3_matches_RY():
    """U3(theta, 0, 0) should equal RY(theta)."""
    theta = math.pi / 2
    params = torch.tensor([theta, 0.0, 0.0])
    u3_mat = U3(params)
    ry_mat = RY(theta)
    assert torch.allclose(u3_mat, ry_mat, atol=1e-5)


def test_U3_is_unitary():
    params = torch.tensor([0.5, 1.2, -0.3])
    mat = U3(params)
    assert _is_unitary(mat)


# --- ZZ ---

def test_ZZ_at_zero_is_identity():
    mat = ZZ(0.0)
    expected = torch.eye(4, dtype=torch.cfloat)
    assert torch.allclose(mat, expected, atol=1e-6)


@pytest.mark.parametrize("theta", [0.5, math.pi / 4, math.pi])
def test_ZZ_is_unitary(theta):
    assert _is_unitary(ZZ(theta))


# --- ZZ_SWAP = SWAP @ ZZ ---

@pytest.mark.parametrize("theta", [0.3, math.pi / 4, math.pi, 1.7])
def test_ZZ_SWAP_equals_SWAP_times_ZZ(theta):
    expected = SWAP() @ ZZ(theta)
    actual = ZZ_SWAP(theta)
    assert torch.allclose(actual, expected, atol=1e-5)


# --- dtype ---

@pytest.mark.parametrize("gate_fn", [I, H, X, Y, Z, CNOT, SWAP])
def test_fixed_gates_dtype(gate_fn):
    assert gate_fn().dtype == torch.cfloat


@pytest.mark.parametrize("gate_fn", [RX, RY, RZ])
def test_parameterized_gates_dtype(gate_fn):
    assert gate_fn(0.5).dtype == torch.cfloat


def test_ZZ_dtype():
    assert ZZ(0.5).dtype == torch.cfloat


def test_ZZ_SWAP_dtype():
    assert ZZ_SWAP(0.5).dtype == torch.cfloat


def test_U3_dtype():
    assert U3(torch.tensor([0.1, 0.2, 0.3])).dtype == torch.cfloat


# --- Shape checks ---

@pytest.mark.parametrize("gate_fn", [I, H, X, Y, Z])
def test_single_qubit_gate_shape(gate_fn):
    assert gate_fn().shape == (2, 2)


@pytest.mark.parametrize("gate_fn", [CNOT, SWAP])
def test_two_qubit_gate_shape(gate_fn):
    assert gate_fn().shape == (4, 4)


@pytest.mark.parametrize("gate_fn", [RX, RY, RZ])
def test_parameterized_gate_shape(gate_fn):
    assert gate_fn(0.5).shape == (2, 2)


def test_U3_shape():
    assert U3(torch.tensor([0.1, 0.2, 0.3])).shape == (2, 2)


def test_ZZ_shape():
    assert ZZ(0.5).shape == (4, 4)


def test_ZZ_SWAP_shape():
    assert ZZ_SWAP(0.5).shape == (4, 4)


# --- Pauli algebra: X^2 = Y^2 = Z^2 = I ---

def test_X_squared_is_identity():
    assert torch.allclose(X() @ X(), I(), atol=1e-6)


def test_Y_squared_is_identity():
    assert torch.allclose(Y() @ Y(), I(), atol=1e-6)


def test_Z_squared_is_identity():
    assert torch.allclose(Z() @ Z(), I(), atol=1e-6)


# --- Pauli algebra: XY = iZ, YZ = iX, ZX = iY ---

def test_XY_equals_iZ():
    assert torch.allclose(X() @ Y(), 1j * Z(), atol=1e-6)


def test_YZ_equals_iX():
    assert torch.allclose(Y() @ Z(), 1j * X(), atol=1e-6)


def test_ZX_equals_iY():
    assert torch.allclose(Z() @ X(), 1j * Y(), atol=1e-6)


# --- Hadamard properties ---

def test_H_squared_is_identity():
    assert torch.allclose(H() @ H(), I(), atol=1e-6)


def test_H_X_H_equals_Z():
    """HXH = Z (Hadamard conjugates X to Z)."""
    h = H()
    assert torch.allclose(h @ X() @ h, Z(), atol=1e-5)


def test_H_Z_H_equals_X():
    """HZH = X (Hadamard conjugates Z to X)."""
    h = H()
    assert torch.allclose(h @ Z() @ h, X(), atol=1e-5)


# --- SWAP properties ---

def test_SWAP_squared_is_identity():
    assert torch.allclose(SWAP() @ SWAP(), torch.eye(4, dtype=torch.cfloat), atol=1e-6)


def test_CNOT_squared_is_identity():
    assert torch.allclose(CNOT() @ CNOT(), torch.eye(4, dtype=torch.cfloat), atol=1e-6)


# --- Parameterized gates: periodicity ---

@pytest.mark.parametrize("gate_fn", [RX, RY, RZ])
def test_rotation_periodicity_4pi(gate_fn):
    """R(4*pi) = I for all rotation gates."""
    mat = gate_fn(4 * math.pi)
    assert torch.allclose(mat, I(), atol=1e-4)


@pytest.mark.parametrize("gate_fn", [RX, RY, RZ])
def test_rotation_at_2pi_is_minus_identity(gate_fn):
    """R(2*pi) = -I for all rotation gates."""
    mat = gate_fn(2 * math.pi)
    assert torch.allclose(mat, -I(), atol=1e-4)


# --- Negative angles ---

@pytest.mark.parametrize("gate_fn", [RX, RY, RZ])
def test_negative_angle_is_inverse(gate_fn):
    """R(-theta) = R(theta)^dagger for rotation gates."""
    theta = 0.7
    forward = gate_fn(theta)
    backward = gate_fn(-theta)
    assert torch.allclose(forward @ backward, I(), atol=1e-5)


# --- Torch tensor input for parameterized gates ---

@pytest.mark.parametrize("gate_fn", [RX, RY, RZ])
def test_parameterized_gate_accepts_torch_tensor(gate_fn):
    theta = torch.tensor(0.5)
    mat = gate_fn(theta)
    assert mat.shape == (2, 2)
    assert _is_unitary(mat)


def test_ZZ_accepts_torch_tensor():
    theta = torch.tensor(0.5)
    mat = ZZ(theta)
    assert mat.shape == (4, 4)
    assert _is_unitary(mat)


# --- U3 special cases ---

def test_U3_matches_RX():
    """U3(theta, -pi/2, pi/2) = RX(theta) up to global phase."""
    theta = 0.7
    params = torch.tensor([theta, -math.pi / 2, math.pi / 2])
    u3_mat = U3(params)
    rx_mat = RX(theta)
    # Compare up to global phase: |det(U3)| should be 1
    assert _is_unitary(u3_mat)
    # Check they produce same action on |0>: same probabilities
    ket0 = torch.tensor([1, 0], dtype=torch.cfloat)
    out_u3 = u3_mat @ ket0
    out_rx = rx_mat @ ket0
    assert torch.allclose(torch.abs(out_u3), torch.abs(out_rx), atol=1e-5)


def test_U3_identity_at_zero():
    """U3(0, 0, 0) = I."""
    params = torch.tensor([0.0, 0.0, 0.0])
    assert torch.allclose(U3(params), I(), atol=1e-6)


@pytest.mark.parametrize("params", [
    torch.tensor([1.0, 2.0, 3.0]),
    torch.tensor([0.0, math.pi, 0.0]),
    torch.tensor([math.pi, math.pi / 2, -math.pi / 4]),
])
def test_U3_is_unitary_various(params):
    assert _is_unitary(U3(params))


# --- ZZ known diagonal values ---

def test_ZZ_diagonal_structure():
    """ZZ(theta) should be diagonal."""
    theta = 0.8
    mat = ZZ(theta)
    off_diag = mat - torch.diag(torch.diag(mat))
    assert torch.allclose(off_diag, torch.zeros(4, 4, dtype=torch.cfloat), atol=1e-6)


def test_ZZ_diagonal_values():
    """Check ZZ(theta) diagonal: (e^{-it/2}, e^{it/2}, e^{it/2}, e^{-it/2})."""
    theta = 1.3
    mat = ZZ(theta)
    a = torch.exp(torch.tensor(-1j * theta / 2))
    b = torch.exp(torch.tensor(1j * theta / 2))
    expected_diag = torch.tensor([a, b, b, a], dtype=torch.cfloat)
    assert torch.allclose(torch.diag(mat), expected_diag, atol=1e-5)


# --- ZZ_SWAP is unitary ---

@pytest.mark.parametrize("theta", [0.0, 0.5, math.pi / 4, math.pi, 2.5])
def test_ZZ_SWAP_is_unitary(theta):
    assert _is_unitary(ZZ_SWAP(theta))


# --- Gate action on basis states ---

def test_X_flips_zero_to_one():
    ket0 = torch.tensor([1, 0], dtype=torch.cfloat)
    ket1 = torch.tensor([0, 1], dtype=torch.cfloat)
    assert torch.allclose(X() @ ket0, ket1)


def test_X_flips_one_to_zero():
    ket0 = torch.tensor([1, 0], dtype=torch.cfloat)
    ket1 = torch.tensor([0, 1], dtype=torch.cfloat)
    assert torch.allclose(X() @ ket1, ket0)


def test_Z_leaves_zero_unchanged():
    ket0 = torch.tensor([1, 0], dtype=torch.cfloat)
    assert torch.allclose(Z() @ ket0, ket0)


def test_Z_flips_phase_of_one():
    ket1 = torch.tensor([0, 1], dtype=torch.cfloat)
    assert torch.allclose(Z() @ ket1, -ket1)


def test_H_on_zero_gives_plus():
    ket0 = torch.tensor([1, 0], dtype=torch.cfloat)
    plus = torch.tensor([1, 1], dtype=torch.cfloat) / math.sqrt(2)
    assert torch.allclose(H() @ ket0, plus, atol=1e-6)


def test_H_on_one_gives_minus():
    ket1 = torch.tensor([0, 1], dtype=torch.cfloat)
    minus = torch.tensor([1, -1], dtype=torch.cfloat) / math.sqrt(2)
    assert torch.allclose(H() @ ket1, minus, atol=1e-6)


def test_CNOT_action_on_basis_states():
    """CNOT: |00>->|00>, |01>->|01>, |10>->|11>, |11>->|10>."""
    cnot = CNOT()
    states = torch.eye(4, dtype=torch.cfloat)
    expected_indices = [0, 1, 3, 2]
    for i, j in enumerate(expected_indices):
        result = cnot @ states[i]
        assert torch.allclose(result, states[j], atol=1e-6), f"|{i:02b}> failed"


def test_SWAP_action_on_basis_states():
    """SWAP: |00>->|00>, |01>->|10>, |10>->|01>, |11>->|11>."""
    swap = SWAP()
    states = torch.eye(4, dtype=torch.cfloat)
    expected_indices = [0, 2, 1, 3]
    for i, j in enumerate(expected_indices):
        result = swap @ states[i]
        assert torch.allclose(result, states[j], atol=1e-6), f"|{i:02b}> failed"
