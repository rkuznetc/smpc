"""Логистическая регрессия с приближённым градиентом (логика ноутбуков 04 и 05).

Сигмоида заменяется линейным приближением Тейлора в нуле:
    σ̃(z) = 0.5 + 0.25·z   (вместо σ(z) = 1 / (1 + e^(−z)))
Тогда остаток равен e = σ̃(z) − y, а градиент:
    ∂/∂w = (1/n) · Xᵀ e,   ∂/∂b = (1/n) · Σ e.
Шифрование CKKS умеет только складывать и умножать, поэтому именно такой
градиент можно вычислить над шифротекстами. Это НЕ градиент log-loss:
в точке равновесия решается МНК на метках, а не максимум правдоподобия.

Обучение возобновляемое: ApproxState хранит веса, bias, номер эпохи и
историю; run_epochs продолжает обучение с сохранённого состояния.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

SIGMOID_SLOPE = 0.25
SIGMOID_OFFSET = 0.5

#: gradient(w, b, X, y) -> (grad_w, grad_b)
GradientFn = Callable[[np.ndarray, float, Any, np.ndarray], tuple[np.ndarray, float]]


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Настоящая сигмоида; используется для вывода вероятностей."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def linear_sigmoid(z: np.ndarray) -> np.ndarray:
    """Линейное приближение σ̃(z) = 0.5 + 0.25·z, то, что реально используется в обучении."""
    return SIGMOID_OFFSET + SIGMOID_SLOPE * z


def initial_bias(y: np.ndarray) -> float:
    """Начальный bias, как в ноутбуках: 4·(mean(y) − 0.5)."""
    return float(4.0 * (np.mean(y) - 0.5))


def plain_gradient(w: np.ndarray, b: float, X: Any, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Открытый приближённый градиент: эталон для проверки CKKS."""
    z = np.asarray(X @ w).ravel() + b
    residual = linear_sigmoid(z) - np.asarray(y, dtype=float)
    grad_w = np.asarray(X.T @ residual).ravel() / X.shape[0]
    return grad_w, float(np.mean(residual))


@dataclass
class ApproxState:
    """Состояние обучения: его достаточно, чтобы продолжить с той же эпохи."""

    weights: np.ndarray
    bias: float
    epoch: int = 0
    history: list[dict[str, float]] = field(default_factory=list)

    @classmethod
    def start(cls, y: np.ndarray, n_features: int) -> "ApproxState":
        return cls(weights=np.zeros(n_features, dtype=float), bias=initial_bias(y))


def run_epochs(
    state: ApproxState,
    X: Any,
    y: np.ndarray,
    gradient: GradientFn,
    epochs: int,
    learning_rate: float = 0.15,
    reg_l2: float = 1e-3,
    on_epoch: Callable[[ApproxState], None] | None = None,
) -> ApproxState:
    """Полный градиентный спуск до эпохи `epochs` (включительно).

    Если state.epoch уже ≥ epochs, ничего не делает. on_epoch вызывается
    после каждой эпохи (например, для сохранения контрольной точки).
    """
    y = np.asarray(y, dtype=float)
    while state.epoch < epochs:
        tic = time.perf_counter()
        grad_w, grad_b = gradient(state.weights, state.bias, X, y)
        state.weights = state.weights - learning_rate * (grad_w + reg_l2 * state.weights)
        state.bias = state.bias - learning_rate * grad_b
        state.epoch += 1
        state.history.append({
            "epoch": float(state.epoch),
            "seconds": time.perf_counter() - tic,
            "grad_norm": float(np.linalg.norm(grad_w)),
        })
        if on_epoch is not None:
            on_epoch(state)
    return state
