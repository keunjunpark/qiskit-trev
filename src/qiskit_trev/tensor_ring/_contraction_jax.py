"""JAX gate-contraction helpers — parallel to :mod:`contraction`.

Uses ``jnp.einsum`` + ``jnp.linalg.svd`` to apply one- and two-qubit gates
to (batches of) tensor-ring cores. Keeps semantics equivalent to the torch
path; numerical agreement requires ``jax.lax.Precision.HIGHEST`` on GPU.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


_JNP_COMPLEX = jnp.complex64


def _einsum(*args, precision):
    return jnp.einsum(*args, precision=precision)


def apply_single_qubit_gate_batch_jax(
    gate_matrix_batch: jnp.ndarray,
    core_batch: jnp.ndarray,
    *,
    precision: jax.lax.Precision = jax.lax.Precision.DEFAULT,
) -> jnp.ndarray:
    """Apply single-qubit gates to a batch of tensor ring cores.

    Args:
        gate_matrix_batch: (B, 2, 2) batch of gate matrices.
        core_batch: (B, chi1, chi2, 2) batch of cores.

    Returns:
        (B, chi1, chi2, 2) updated batch of cores.
    """
    result = _einsum("bij,bklj->bikl", gate_matrix_batch, core_batch, precision=precision)
    return jnp.transpose(result, (0, 2, 3, 1))


def apply_double_qubit_gate_batch_jax(
    gate_matrix: jnp.ndarray,
    core_a_batch: jnp.ndarray,
    core_b_batch: jnp.ndarray,
    *,
    max_rank: int | None = None,
    precision: jax.lax.Precision = jax.lax.Precision.DEFAULT,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply a two-qubit gate to a batch of adjacent tensor ring core pairs.

    Args:
        gate_matrix: (4, 4) or (B, 4, 4) gate matrix.
        core_a_batch: (B, chi1, chi2, 2).
        core_b_batch: (B, chi2, chi3, 2).
        max_rank: Max bond dim after SVD truncation. Defaults to min(chi1, chi3).
    """
    B, chi1, chi2, _ = core_a_batch.shape
    _, chi2_, chi3, _ = core_b_batch.shape
    assert chi2 == chi2_, "Bond mismatch between the two site tensors"

    if max_rank is None:
        max_rank = min(chi1, chi3)

    mps = _einsum("bikp,bkjq->bijpq", core_a_batch, core_b_batch, precision=precision)

    g = gate_matrix
    if g.ndim == 2:
        g = jnp.broadcast_to(g, (B,) + g.shape)

    mps = mps.reshape(B, chi1 * chi3, 4)
    mps = jnp.matmul(mps, jnp.transpose(g, (0, 2, 1)), precision=precision)
    mps = mps.reshape(B, chi1, chi3, 2, 2)
    mps = jnp.transpose(mps, (0, 3, 1, 4, 2)).reshape(B, 2 * chi1, 2 * chi3)

    u, s, vh = jnp.linalg.svd(mps, full_matrices=False)
    k = min(max_rank, s.shape[-1])
    x = u[:, :, :k]
    sx = (s[:, :k, None] * jnp.eye(k, dtype=s.dtype)).astype(_JNP_COMPLEX)
    y = vh[:, :k, :]

    new_a = jnp.matmul(x, sx, precision=precision).reshape(B, 2, chi1, k)
    new_a = jnp.transpose(new_a, (0, 2, 3, 1))
    new_b = y.reshape(B, k, 2, chi3)
    new_b = jnp.transpose(new_b, (0, 1, 3, 2))

    return new_a, new_b


def swap_gate_matrix_jax(matrix: jnp.ndarray) -> jnp.ndarray:
    """Permute qubit labels in a 4x4 gate: SWAP @ matrix @ SWAP.

    Accepts (4, 4) or (B, 4, 4).
    """
    swap = jnp.asarray(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=matrix.dtype,
    )
    if matrix.ndim == 2:
        return swap @ matrix @ swap
    # batched: apply on last two dims
    return jnp.einsum("ij,bjk,kl->bil", swap, matrix, swap)
