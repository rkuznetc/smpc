import numpy as np
import pytest
import scipy.sparse as sp

pytest.importorskip("tenseal")

from smpc.he.ckks import CKKSGradient, DemoAggregateDecryptor, make_context_pair  # noqa: E402
from smpc.ml.approx import plain_gradient  # noqa: E402


@pytest.fixture(scope="module")
def contexts():
    return make_context_pair()


def _data(seed=0, n=40, p=4):
    rng = np.random.default_rng(seed)
    X = sp.csr_matrix(rng.uniform(-1, 1, size=(n, p)).astype(np.float32))
    y = rng.integers(0, 2, n).astype(float)
    w = rng.normal(scale=0.1, size=p)
    return X, y, w, 0.2


def test_public_context_has_no_secret_key(contexts):
    _, public = contexts
    assert not public.has_secret_key()


def test_ckks_gradient_matches_plain_approximation(contexts):
    private, public = contexts
    X, y, w, b = _data()
    gradient = CKKSGradient(public, DemoAggregateDecryptor(private), batch_size=16)
    g_he, b_he = gradient(w, b, X, y)
    g_plain, b_plain = plain_gradient(w, b, X, y)
    assert g_he == pytest.approx(g_plain, abs=1e-3)
    assert b_he == pytest.approx(b_plain, abs=1e-3)


def test_batch_size_does_not_change_the_gradient(contexts):
    private, public = contexts
    X, y, w, b = _data(seed=1)
    decrypt = DemoAggregateDecryptor(private)
    small = CKKSGradient(public, decrypt, batch_size=7)(w, b, X, y)
    whole = CKKSGradient(public, decrypt, batch_size=40)(w, b, X, y)
    assert small[0] == pytest.approx(whole[0], abs=1e-3)
    assert small[1] == pytest.approx(whole[1], abs=1e-3)


def test_masked_decryptor_returns_original_value(contexts):
    import tenseal as ts

    private, public = contexts
    encrypted = ts.ckks_vector(public, [3.5, 0.0]).sum()
    assert DemoAggregateDecryptor(private)(encrypted) == pytest.approx(3.5, abs=1e-3)
