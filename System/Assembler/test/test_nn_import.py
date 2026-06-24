"""Tests for depth-independent Neural_Networks resolution and NN_import."""

import numpy as np
import pytest

from nn_assembler.Convert import NETWORKS_DIR_ENV, NN_import, find_networks_dir


def _make_network(nn_root, model="Tiny_NN", run="Recent"):
    model_dir = nn_root / model
    model_dir.mkdir(parents=True)
    (model_dir / f"{model}_{run}.mlir").write_text("module @m {}\n")
    np.savez(model_dir / f"{model}_{run}.weights.npz", w=np.zeros(1, dtype=np.float32))


def test_find_networks_dir_via_env(tmp_path, monkeypatch):
    nn_root = tmp_path / "Neural_Networks"
    nn_root.mkdir()
    monkeypatch.setenv(NETWORKS_DIR_ENV, str(nn_root))
    assert find_networks_dir() == nn_root


def test_find_networks_dir_env_missing_dir_errors(tmp_path, monkeypatch):
    monkeypatch.setenv(NETWORKS_DIR_ENV, str(tmp_path / "does_not_exist"))
    with pytest.raises(AssertionError):
        find_networks_dir()


def test_nn_import_accepts_explicit_dir(tmp_path):
    nn_root = tmp_path / "Neural_Networks"
    _make_network(nn_root)
    work = tmp_path / "tmp"

    NN_import("Tiny_NN", "Recent", tmp_dir=work, nn_dir=nn_root)

    assert (work / "initial.mlir").is_file()
    assert (work / "weights.npz").is_file()


def test_nn_import_missing_network_is_descriptive(tmp_path):
    nn_root = tmp_path / "Neural_Networks"
    nn_root.mkdir()
    with pytest.raises(AssertionError, match="Missing network graph"):
        NN_import("Nope", "Recent", tmp_dir=tmp_path / "tmp", nn_dir=nn_root)
