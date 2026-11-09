import numpy as np
import pytest
import scipy.sparse as sp

from smpc.ml.approx import ApproxState
from smpc.ml.checkpoint import CheckpointMismatch, fingerprint, load_state, save_state

NAMES = ["bank__a", "bank__b"]
CONFIG = {"batch_size": 8, "learning_rate": 0.15, "reg_l2": 1e-3}


def _X():
    return sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]]))


def test_roundtrip_keeps_state(tmp_path):
    state = ApproxState(weights=np.array([1.5, -2.0]), bias=0.25, epoch=2,
                        history=[{"epoch": 1.0, "seconds": 0.1, "grad_norm": 0.3}])
    fp = fingerprint(_X(), np.array([0.0, 1.0]), NAMES, CONFIG)
    path = tmp_path / "ckpt" / "run.npz"
    save_state(path, state, fp, NAMES)
    loaded = load_state(path, fp)
    assert loaded.epoch == 2
    assert loaded.weights == pytest.approx(state.weights)
    assert loaded.bias == pytest.approx(0.25)
    assert loaded.history == state.history
    assert not list((tmp_path / "ckpt").glob("*.tmp.npz"))


def test_missing_checkpoint_returns_none(tmp_path):
    assert load_state(tmp_path / "none.npz", "anything") is None


def test_fingerprint_mismatch_is_an_error(tmp_path):
    state = ApproxState(weights=np.zeros(2), bias=0.0, epoch=1)
    fp = fingerprint(_X(), np.array([0.0, 1.0]), NAMES, CONFIG)
    path = tmp_path / "run.npz"
    save_state(path, state, fp, NAMES)
    other = fingerprint(_X(), np.array([1.0, 0.0]), NAMES, CONFIG)
    with pytest.raises(CheckpointMismatch):
        load_state(path, other)


def test_fingerprint_depends_on_data_names_and_config():
    y = np.array([0.0, 1.0])
    base = fingerprint(_X(), y, NAMES, CONFIG)
    assert base == fingerprint(_X(), y, NAMES, CONFIG)
    assert base != fingerprint(_X(), np.array([1.0, 1.0]), NAMES, CONFIG)
    assert base != fingerprint(_X(), y, ["x", "y"], CONFIG)
    assert base != fingerprint(_X(), y, NAMES, {**CONFIG, "learning_rate": 0.2})
