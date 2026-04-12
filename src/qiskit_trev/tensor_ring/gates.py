"""Gate matrix functions for tensor ring simulation.

Ported from TREV gates/info.py (real_form_autograd branch).
All functions return torch.cfloat tensors.
"""

import math

import torch
from torch import Tensor


def I(device: str = None) -> Tensor:
    return torch.eye(2, dtype=torch.cfloat, device=device)


def X(device: str = None) -> Tensor:
    return torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat, device=device)


def Y(device: str = None) -> Tensor:
    return torch.tensor([[0, -1j], [1j, 0]], dtype=torch.cfloat, device=device)


def Z(device: str = None) -> Tensor:
    return torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat, device=device)


def H(device: str = None) -> Tensor:
    return (1 / math.sqrt(2)) * torch.tensor(
        [[1, 1], [1, -1]], dtype=torch.cfloat, device=device
    )


def RX(theta, device: str = None) -> Tensor:
    is_scalar = not isinstance(theta, torch.Tensor) or theta.dim() == 0
    theta = torch.atleast_1d(torch.as_tensor(theta, dtype=torch.float))
    cos = torch.cos(theta / 2)
    sin = torch.sin(theta / 2)
    rx = torch.stack([
        torch.stack([cos, -1j * sin], dim=-1),
        torch.stack([-1j * sin, cos], dim=-1),
    ], dim=-2).to(device=device, dtype=torch.cfloat)
    return rx[0] if is_scalar else rx


def RY(theta, device: str = None) -> Tensor:
    is_scalar = not isinstance(theta, torch.Tensor) or theta.dim() == 0
    theta = torch.atleast_1d(torch.as_tensor(theta, dtype=torch.float))
    cos = torch.cos(theta / 2)
    sin = torch.sin(theta / 2)
    ry = torch.stack([
        torch.stack([cos, -sin], dim=-1),
        torch.stack([sin, cos], dim=-1),
    ], dim=-2).to(device=device, dtype=torch.cfloat)
    return ry[0] if is_scalar else ry


def RZ(theta, device: str = None) -> Tensor:
    is_scalar = not isinstance(theta, torch.Tensor) or theta.dim() == 0
    theta = torch.atleast_1d(torch.as_tensor(theta, dtype=torch.float))
    exp_m = torch.exp(-1j * theta / 2)
    exp_p = torch.exp(1j * theta / 2)
    rz = torch.stack([
        torch.stack([exp_m, torch.zeros_like(theta)], dim=-1),
        torch.stack([torch.zeros_like(theta), exp_p], dim=-1),
    ], dim=-2).to(device=device, dtype=torch.cfloat)
    return rz[0] if is_scalar else rz


def U3(params: Tensor, device: str = None) -> Tensor:
    """U3(theta, phi, lam) = Rz(phi) . Ry(theta) . Rz(lam).

    Args:
        params: shape (3,) for scalar or (batch, 3) for batch.
    """
    is_scalar = params.dim() == 1
    if is_scalar:
        params = params.unsqueeze(0)
    theta = params[:, 0]
    phi = params[:, 1]
    lam = params[:, 2]
    cos = torch.cos(theta / 2)
    sin = torch.sin(theta / 2)
    u00 = torch.exp(-1j * (phi + lam) / 2) * cos
    u01 = -torch.exp(-1j * (phi - lam) / 2) * sin
    u10 = torch.exp(1j * (phi - lam) / 2) * sin
    u11 = torch.exp(1j * (phi + lam) / 2) * cos
    mat = torch.stack([
        torch.stack([u00, u01], dim=-1),
        torch.stack([u10, u11], dim=-1),
    ], dim=-2).to(device=device, dtype=torch.cfloat)
    return mat[0] if is_scalar else mat


def CNOT(device: str = None) -> Tensor:
    return torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ], dtype=torch.cfloat, device=device)


def SWAP(device: str = None) -> Tensor:
    return torch.tensor([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=torch.cfloat, device=device)


def ZZ(theta, device: str = None) -> Tensor:
    """ZZ(theta) = diag(e^{-i*theta/2}, e^{i*theta/2}, e^{i*theta/2}, e^{-i*theta/2})."""
    is_scalar = not isinstance(theta, torch.Tensor) or theta.dim() == 0
    theta = torch.atleast_1d(torch.as_tensor(theta, dtype=torch.float))
    a = torch.exp(-1j * theta / 2)
    b = torch.exp(1j * theta / 2)
    z = torch.zeros_like(theta)
    row0 = torch.stack([a, z, z, z], dim=-1)
    row1 = torch.stack([z, b, z, z], dim=-1)
    row2 = torch.stack([z, z, b, z], dim=-1)
    row3 = torch.stack([z, z, z, a], dim=-1)
    mat = torch.stack([row0, row1, row2, row3], dim=-2).to(
        dtype=torch.cfloat, device=device
    )
    return mat[0] if is_scalar else mat


def ZZ_SWAP(theta, device: str = None) -> Tensor:
    """ZZ_SWAP(theta) = SWAP . ZZ(theta)."""
    is_scalar = not isinstance(theta, torch.Tensor) or theta.dim() == 0
    theta = torch.atleast_1d(torch.as_tensor(theta, dtype=torch.float))
    a = torch.exp(-1j * theta / 2)
    b = torch.exp(1j * theta / 2)
    z = torch.zeros_like(theta)
    row0 = torch.stack([a, z, z, z], dim=-1)
    row1 = torch.stack([z, z, b, z], dim=-1)
    row2 = torch.stack([z, b, z, z], dim=-1)
    row3 = torch.stack([z, z, z, a], dim=-1)
    mat = torch.stack([row0, row1, row2, row3], dim=-2).to(
        dtype=torch.cfloat, device=device
    )
    return mat[0] if is_scalar else mat
