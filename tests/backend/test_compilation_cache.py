"""Smoke test for enable_compilation_cache helper."""

from __future__ import annotations

import os

import pytest

jax = pytest.importorskip("jax")

from qiskit_trev.backend import enable_compilation_cache


def test_enable_compilation_cache_creates_dir(tmp_path):
    target = tmp_path / "my_cache"
    assert not target.exists()
    resolved = enable_compilation_cache(str(target))
    assert resolved == str(target)
    assert target.exists() and target.is_dir()


def test_enable_compilation_cache_default_path(tmp_path, monkeypatch):
    # Redirect HOME so the default ~/.cache path lands in tmp_path.
    monkeypatch.setenv("HOME", str(tmp_path))
    resolved = enable_compilation_cache()
    assert resolved == os.path.expanduser("~/.cache/qiskit_trev_jax")
    assert os.path.isdir(resolved)
