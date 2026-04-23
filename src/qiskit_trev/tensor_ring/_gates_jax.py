"""JAX gate templates — parallel to :mod:`gates`.

Mirrors the API of ``gates.py`` but returns ``jax.Array``. Used by the JAX
path of :meth:`TensorRingState.build_batch`. Parameterized rotations accept
both scalar and batched ``theta``.
"""

from __future__ import annotations

import math

import jax.numpy as jnp


def I(device=None):
    return jnp.eye(2, dtype=jnp.complex64)


def X(device=None):
    return jnp.asarray([[0, 1], [1, 0]], dtype=jnp.complex64)


def Y(device=None):
    return jnp.asarray([[0, -1j], [1j, 0]], dtype=jnp.complex64)


def Z(device=None):
    return jnp.asarray([[1, 0], [0, -1]], dtype=jnp.complex64)


def H(device=None):
    s = 1.0 / math.sqrt(2.0)
    return jnp.asarray([[s, s], [s, -s]], dtype=jnp.complex64)


def _is_scalar(theta):
    if isinstance(theta, jnp.ndarray):
        return theta.ndim == 0
    return not hasattr(theta, "ndim") or getattr(theta, "ndim", 0) == 0


def RX(theta, device=None):
    is_scalar = _is_scalar(theta)
    theta_arr = jnp.atleast_1d(jnp.asarray(theta, dtype=jnp.float32))
    cos = jnp.cos(theta_arr / 2)
    sin = jnp.sin(theta_arr / 2)
    rx = jnp.stack(
        [
            jnp.stack([cos, -1j * sin], axis=-1),
            jnp.stack([-1j * sin, cos], axis=-1),
        ],
        axis=-2,
    ).astype(jnp.complex64)
    return rx[0] if is_scalar else rx


def RY(theta, device=None):
    is_scalar = _is_scalar(theta)
    theta_arr = jnp.atleast_1d(jnp.asarray(theta, dtype=jnp.float32))
    cos = jnp.cos(theta_arr / 2)
    sin = jnp.sin(theta_arr / 2)
    ry = jnp.stack(
        [
            jnp.stack([cos, -sin], axis=-1),
            jnp.stack([sin, cos], axis=-1),
        ],
        axis=-2,
    ).astype(jnp.complex64)
    return ry[0] if is_scalar else ry


def RZ(theta, device=None):
    is_scalar = _is_scalar(theta)
    theta_arr = jnp.atleast_1d(jnp.asarray(theta, dtype=jnp.float32))
    exp_m = jnp.exp(-1j * theta_arr / 2)
    exp_p = jnp.exp(1j * theta_arr / 2)
    zero = jnp.zeros_like(exp_m)
    rz = jnp.stack(
        [
            jnp.stack([exp_m, zero], axis=-1),
            jnp.stack([zero, exp_p], axis=-1),
        ],
        axis=-2,
    ).astype(jnp.complex64)
    return rz[0] if is_scalar else rz


def U3(params, device=None):
    """U3(theta, phi, lam) = Rz(phi) . Ry(theta) . Rz(lam).

    Args:
        params: shape (3,) scalar or (batch, 3) batched.
    """
    params = jnp.asarray(params)
    is_scalar = params.ndim == 1
    if is_scalar:
        params = params[None, :]
    theta = params[:, 0]
    phi = params[:, 1]
    lam = params[:, 2]
    cos = jnp.cos(theta / 2)
    sin = jnp.sin(theta / 2)
    u00 = jnp.exp(-1j * (phi + lam) / 2) * cos
    u01 = -jnp.exp(-1j * (phi - lam) / 2) * sin
    u10 = jnp.exp(1j * (phi - lam) / 2) * sin
    u11 = jnp.exp(1j * (phi + lam) / 2) * cos
    mat = jnp.stack(
        [
            jnp.stack([u00, u01], axis=-1),
            jnp.stack([u10, u11], axis=-1),
        ],
        axis=-2,
    ).astype(jnp.complex64)
    return mat[0] if is_scalar else mat


def CNOT(device=None):
    return jnp.asarray(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=jnp.complex64,
    )


def SWAP(device=None):
    return jnp.asarray(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=jnp.complex64,
    )


def ZZ(theta, device=None):
    is_scalar = _is_scalar(theta)
    theta_arr = jnp.atleast_1d(jnp.asarray(theta, dtype=jnp.float32))
    a = jnp.exp(-1j * theta_arr / 2)
    b = jnp.exp(1j * theta_arr / 2)
    z = jnp.zeros_like(theta_arr, dtype=jnp.complex64)
    a_ = a.astype(jnp.complex64)
    b_ = b.astype(jnp.complex64)
    row0 = jnp.stack([a_, z, z, z], axis=-1)
    row1 = jnp.stack([z, b_, z, z], axis=-1)
    row2 = jnp.stack([z, z, b_, z], axis=-1)
    row3 = jnp.stack([z, z, z, a_], axis=-1)
    mat = jnp.stack([row0, row1, row2, row3], axis=-2)
    return mat[0] if is_scalar else mat


def ZZ_SWAP(theta, device=None):
    is_scalar = _is_scalar(theta)
    theta_arr = jnp.atleast_1d(jnp.asarray(theta, dtype=jnp.float32))
    a = jnp.exp(-1j * theta_arr / 2)
    b = jnp.exp(1j * theta_arr / 2)
    z = jnp.zeros_like(theta_arr, dtype=jnp.complex64)
    a_ = a.astype(jnp.complex64)
    b_ = b.astype(jnp.complex64)
    row0 = jnp.stack([a_, z, z, z], axis=-1)
    row1 = jnp.stack([z, z, b_, z], axis=-1)
    row2 = jnp.stack([z, b_, z, z], axis=-1)
    row3 = jnp.stack([z, z, z, a_], axis=-1)
    mat = jnp.stack([row0, row1, row2, row3], axis=-2)
    return mat[0] if is_scalar else mat
