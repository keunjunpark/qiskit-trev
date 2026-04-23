"""JAX compute kernels for :class:`qiskit_trev.qml.QMLModel`.

Parallel to the torch methods `_measure_all_qubits`, `_build_param_batch`,
`_build_mega_batch` — same math, jnp ops. Public API lives on QMLModel;
this module is an implementation detail.

The core trick (shared with the torch path) is the ``(Q, B, χ, χ, χ, χ)``
"multi-observable" tensor that carries one slot per qubit-Z observable
through the N-site contraction in a single pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from .tensor_ring._state_jax import build_batch_jax

if TYPE_CHECKING:
    from .tensor_ring.state import GateInstruction


def build_param_batch_jax(
    X: jnp.ndarray,
    theta: jnp.ndarray,
    total_slots: int,
    data_idx: jnp.ndarray,
    feat_idx: jnp.ndarray,
    train_idx: jnp.ndarray,
) -> jnp.ndarray:
    """JAX twin of :meth:`QMLModel._build_param_batch`.

    ``X`` is ``(N, n_features)`` data; ``theta`` is ``(P,)`` trainable.
    Returns ``(N, total_slots)`` params, with data features placed at
    ``data_idx`` (indexed through ``feat_idx``) and trainable params at
    ``train_idx``.
    """
    N = X.shape[0]
    p = jnp.zeros((N, total_slots), dtype=jnp.float32)
    X_j = jnp.asarray(X, dtype=jnp.float32)
    theta_j = jnp.asarray(theta, dtype=jnp.float32)

    # Data slots: p[:, data_idx[i]] = X[:, feat_idx[i]] for each i.
    p = p.at[:, data_idx].set(X_j[:, feat_idx])
    # Trainable slots: broadcast theta across the batch dimension.
    p = p.at[:, train_idx].set(jnp.broadcast_to(theta_j[None, :], (N, theta_j.shape[0])))
    return p


def build_mega_batch_jax(
    X: jnp.ndarray,
    pop_thetas: jnp.ndarray,
    total_slots: int,
    data_idx: jnp.ndarray,
    feat_idx: jnp.ndarray,
    train_idx: jnp.ndarray,
) -> jnp.ndarray:
    """JAX twin of :meth:`QMLModel._build_mega_batch` for populations.

    ``(ps, N) → (ps·N, total_slots)`` with data features repeated across
    population and population thetas broadcast across data samples.
    """
    nd = X.shape[0]
    ps = pop_thetas.shape[0]

    X_j = jnp.asarray(X, dtype=jnp.float32)
    pt_j = jnp.asarray(pop_thetas, dtype=jnp.float32)

    m = jnp.zeros((ps * nd, total_slots), dtype=jnp.float32)
    # Data slots: tile X across the population axis then flatten.
    data_block = jnp.tile(X_j[:, feat_idx], (ps, 1))
    m = m.at[:, data_idx].set(data_block)
    # Trainable slots: broadcast pop_thetas[:, None, :] across data samples,
    # reshape to (ps·N, P) and place at train_idx.
    train_block = jnp.broadcast_to(
        pt_j[:, None, :], (ps, nd, pt_j.shape[1])
    ).reshape(ps * nd, pt_j.shape[1])
    m = m.at[:, train_idx].set(train_block)
    return m


def measure_all_qubits_jax(
    num_qubits: int,
    rank: int,
    gates: list,
    params_batch: jnp.ndarray,
    *,
    precision: jax.lax.Precision = jax.lax.Precision.DEFAULT,
) -> jnp.ndarray:
    """JAX twin of :meth:`QMLModel._measure_all_qubits`.

    Fuses all Q per-qubit Z observables into a single N-site contraction
    by carrying a ``(Q, B, χ, χ, χ, χ)`` tensor where slot ``q`` encodes
    "Z observable on qubit q, identity on the rest".

    Returns ``(Q, B)`` real expectation values.
    """
    Q = num_qubits
    bt = build_batch_jax(Q, rank, gates, params_batch, precision=precision)
    # bt: (B, N, χ, χ, 2)

    # First site: compute E_I and E_Z transfer matrices.
    A = bt[:, 0]
    E_I = jnp.einsum('blrd,bLRd->blLrR', A.conj(), A, precision=precision)
    corr = jnp.einsum(
        'blr,bLR->blLrR', A.conj()[..., 1], A[..., 1], precision=precision
    )
    E_Z = E_I - 2 * corr

    # ten[q] starts with E_Z at slot 0 (observable is Z at site 0 for q=0),
    # E_I elsewhere (observable is I at site 0 for q≠0 — Z moves to site q
    # on the corresponding iteration).
    ten = jnp.broadcast_to(E_I[None], (Q,) + E_I.shape)
    ten = ten.at[0].set(E_Z)

    # Contract remaining sites. At site i: all slots contract with E_I
    # through site i, except slot i which contracts with E_Z (its Z is at
    # site i).
    for i in range(1, Q):
        A = bt[:, i]
        Ei_I = jnp.einsum('blrd,bLRd->blLrR', A.conj(), A, precision=precision)
        corr = jnp.einsum(
            'blr,bLR->blLrR', A.conj()[..., 1], A[..., 1], precision=precision
        )
        Ei_Z = Ei_I - 2 * corr

        ten_i_prev = ten[i]
        ten = jnp.einsum(
            'Qbijpq,bpqrs->Qbijrs', ten, Ei_I, precision=precision
        )
        new_slot = jnp.einsum(
            'bijpq,bpqrs->bijrs', ten_i_prev, Ei_Z, precision=precision
        )
        ten = ten.at[i].set(new_slot)

    # Close ring: trace over (i=r, j=s) → (Q, B) complex, take real part.
    return jnp.einsum('Qbijij->Qb', ten).real


def parameter_shift_grad_jax(
    num_qubits: int,
    rank: int,
    gates: list,
    X: jnp.ndarray,
    theta: jnp.ndarray,
    total_slots: int,
    data_idx: jnp.ndarray,
    feat_idx: jnp.ndarray,
    train_idx: jnp.ndarray,
    shift: float,
    *,
    chunk_size: int | None = None,
    precision: jax.lax.Precision = jax.lax.Precision.DEFAULT,
) -> jnp.ndarray:
    """JAX twin of :meth:`QMLModel.parameter_shift_grad`.

    Builds the base param batch, then for each chunk of P parameters
    constructs the ``(2C, N, total_slots)`` shifted block, reshapes to
    ``(2C·N, total_slots)``, measures, and folds into the gradient.

    Uses jit-compiled per-chunk kernels — the outer chunk loop is
    Python but each kernel compiles once per shape.
    """
    import math

    N = X.shape[0]
    P = int(theta.shape[0])
    Q = num_qubits
    denom = 2.0 * math.sin(shift)

    base = build_param_batch_jax(
        X, theta, total_slots, data_idx, feat_idx, train_idx
    )

    chunk_size = chunk_size if chunk_size is not None else P
    chunk_size = max(1, min(P, chunk_size))

    grad = jnp.zeros((Q, P, N), dtype=jnp.float32)

    for start in range(0, P, chunk_size):
        stop = min(start + chunk_size, P)
        C = stop - start

        chunk_idx = train_idx[start:stop]
        shift_add = jnp.zeros((C, 2, total_slots), dtype=base.dtype)
        rows = jnp.arange(C)
        shift_add = shift_add.at[rows, 0, chunk_idx].set(shift)
        shift_add = shift_add.at[rows, 1, chunk_idx].set(-shift)
        shift_add = shift_add.reshape(2 * C, total_slots)

        blk = (base[None, :, :] + shift_add[:, None, :]).reshape(
            2 * C * N, total_slots
        )
        evs = measure_all_qubits_jax(
            Q, rank, gates, blk, precision=precision
        )
        evs = evs.reshape(Q, C, 2, N)
        grad = grad.at[:, start:stop, :].set(
            (evs[:, :, 0, :] - evs[:, :, 1, :]) / denom
        )

    return grad


def forward_population_jax(
    num_qubits: int,
    rank: int,
    gates: list,
    X: jnp.ndarray,
    pop_thetas: jnp.ndarray,
    total_slots: int,
    data_idx: jnp.ndarray,
    feat_idx: jnp.ndarray,
    train_idx: jnp.ndarray,
    *,
    precision: jax.lax.Precision = jax.lax.Precision.DEFAULT,
) -> jnp.ndarray:
    """JAX twin of :meth:`QMLModel.forward_population`.

    Returns ``(Q, pop_size, N)``.
    """
    ps = int(pop_thetas.shape[0])
    N = int(X.shape[0])

    mega = build_mega_batch_jax(
        X, pop_thetas, total_slots, data_idx, feat_idx, train_idx
    )
    all_evs = measure_all_qubits_jax(
        num_qubits, rank, gates, mega, precision=precision
    )
    return all_evs.reshape(num_qubits, ps, N)
