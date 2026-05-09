"""Smoke tests for enable_compilation_cache helper."""

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
    monkeypatch.delenv("QISKIT_TREV_CACHE_DIR", raising=False)
    resolved = enable_compilation_cache()
    assert resolved == os.path.expanduser("~/.cache/qiskit_trev_jax")
    assert os.path.isdir(resolved)


def test_env_var_overrides_default(tmp_path, monkeypatch):
    # QISKIT_TREV_CACHE_DIR is the platform-agnostic way to redirect the
    # cache (Drive on Colab, $SCRATCH on HPC, etc.).
    target = tmp_path / "env_cache"
    monkeypatch.setenv("QISKIT_TREV_CACHE_DIR", str(target))
    resolved = enable_compilation_cache()
    assert resolved == str(target)
    assert target.is_dir()


def test_explicit_arg_beats_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("QISKIT_TREV_CACHE_DIR", str(tmp_path / "from_env"))
    explicit = tmp_path / "from_arg"
    resolved = enable_compilation_cache(str(explicit))
    assert resolved == str(explicit)
    assert explicit.is_dir()


@pytest.mark.parametrize("level", [0, 2, 4])
def test_autotune_level_sets_xla_flag(level, monkeypatch, tmp_path):
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    enable_compilation_cache(str(tmp_path / "c"), autotune_level=level)
    flags = os.environ.get("XLA_FLAGS", "")
    assert f"--xla_gpu_autotune_level={level}" in flags


def test_autotune_level_replaces_old_value(monkeypatch, tmp_path):
    # Subsequent calls must replace the flag, not duplicate it.
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_autotune_level=4 --foo=bar")
    enable_compilation_cache(str(tmp_path / "c"), autotune_level=0)
    flags = os.environ["XLA_FLAGS"]
    assert flags.count("--xla_gpu_autotune_level=") == 1
    assert "--xla_gpu_autotune_level=0" in flags
    assert "--foo=bar" in flags


def test_autotune_level_validation(tmp_path):
    with pytest.raises(ValueError):
        enable_compilation_cache(str(tmp_path / "c"), autotune_level=1)
    with pytest.raises(ValueError):
        enable_compilation_cache(str(tmp_path / "c"), autotune_level=3)
