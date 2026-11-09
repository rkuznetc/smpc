import numpy as np
import pytest
import scipy.sparse as sp

from smpc.ml.approx import ApproxState, initial_bias, linear_sigmoid, plain_gradient, run_epochs, sigmoid


def test_sigmoid_is_bounded_and_centred():
    assert sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)
    out = sigmoid(np.array([-1000.0, 1000.0]))
    assert np.all((out > 0) & (out < 1))


def test_linear_sigmoid_is_taylor_expansion_at_zero():
    z = np.array([0.0, 0.1, -0.1])
    assert linear_sigmoid(z) == pytest.approx(0.5 + 0.25 * z)
    # В окрестности нуля приближение близко к настоящей сигмоиде.
    assert np.max(np.abs(linear_sigmoid(z) - sigmoid(z))) < 1e-3


def test_initial_bias_formula():
    assert initial_bias(np.array([0, 1, 0, 0])) == pytest.approx(-1.0)  # 4 * (0.25 - 0.5)


def _mean_squared_residual(w, b, X, y):
    z = np.asarray(X @ w).ravel() + b
    return float(np.mean((linear_sigmoid(z) - y) ** 2))


@pytest.mark.parametrize("sparse", [False, True])
def test_plain_gradient_is_twice_gradient_of_mse_of_linear_sigmoid(sparse):
    # Документирует математику: приближённый градиент — МНК на метках,
    # он равен 2 · ∂/∂w mean((σ̃(z) − y)²).
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 4))
    X = sp.csr_matrix(X) if sparse else X
    y = rng.integers(0, 2, 30).astype(float)
    w, b = rng.normal(scale=0.2, size=4), 0.1
    grad_w, grad_b = plain_gradient(w, b, X, y)

    eps = 1e-6
    num_w = np.array([
        (_mean_squared_residual(w + eps * e, b, X, y) - _mean_squared_residual(w - eps * e, b, X, y)) / (2 * eps)
        for e in np.eye(4)
    ])
    num_b = (_mean_squared_residual(w, b + eps, X, y) - _mean_squared_residual(w, b - eps, X, y)) / (2 * eps)
    assert grad_w == pytest.approx(2 * num_w, abs=1e-5)
    assert grad_b == pytest.approx(2 * num_b, abs=1e-5)


def test_gradient_descent_reduces_error():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] > 0).astype(float)
    state = ApproxState.start(y, 3)
    before = _mean_squared_residual(state.weights, state.bias, X, y)
    run_epochs(state, X, y, plain_gradient, epochs=20)
    assert _mean_squared_residual(state.weights, state.bias, X, y) < before


def test_run_epochs_resumes_and_matches_straight_run():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(50, 2))
    y = (X[:, 0] > 0).astype(float)

    straight = run_epochs(ApproxState.start(y, 2), X, y, plain_gradient, epochs=4)

    calls = []
    part = run_epochs(ApproxState.start(y, 2), X, y, plain_gradient, epochs=2,
                      on_epoch=lambda s: calls.append(s.epoch))
    resumed = run_epochs(part, X, y, plain_gradient, epochs=4, on_epoch=lambda s: calls.append(s.epoch))

    assert calls == [1, 2, 3, 4]
    assert resumed.epoch == 4 and len(resumed.history) == 4
    assert resumed.weights == pytest.approx(straight.weights)
    assert resumed.bias == pytest.approx(straight.bias)


def test_run_epochs_is_noop_when_already_trained():
    y = np.array([0.0, 1.0])
    X = np.array([[1.0], [2.0]])
    state = run_epochs(ApproxState.start(y, 1), X, y, plain_gradient, epochs=2)
    again = run_epochs(state, X, y, plain_gradient, epochs=2)
    assert again.epoch == 2 and len(again.history) == 2
