"""Модели скоринга: интерфейс и три сравниваемые модели.

Сравнение (ноутбук 05), на отложенной выборке общих клиентов:
    BankOnlyModel — база: только признаки банка;
    CeilingModel  — признаки банка + все признаки телекома в открытую
                    (только лаборатория; по выходу совпадает с версией на CKKS);
    WoEScorecard  — модель из агрегатов PIS (пока заглушка, см. ниже).

Обучение моделей — приближённый градиент из ml/approx.py. Градиент можно
подменить: по умолчанию открытый (plain), для CKKS — CKKSGradient из smpc.he.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import numpy as np
import pandas as pd

from smpc.ml.approx import ApproxState, GradientFn, plain_gradient, run_epochs, sigmoid


class ScoringModel(ABC):
    """Модель скоринга: вероятность дефолта по признакам клиента."""

    name: str = "model"

    @abstractmethod
    def fit(self, X: Any, y: np.ndarray) -> "ScoringModel":
        """Обучение. Возвращает self."""

    @abstractmethod
    def predict_proba(self, X: Any) -> np.ndarray:
        """Вероятность дефолта (класс 1) для каждой строки X."""


class ApproxLogisticModel(ScoringModel):
    """Логистическая регрессия с приближённым градиентом (см. ml/approx.py)."""

    name = "approx_logistic"

    def __init__(
        self,
        gradient: GradientFn | None = None,
        epochs: int = 10,
        learning_rate: float = 0.15,
        reg_l2: float = 1e-3,
        on_epoch=None,
    ) -> None:
        self.gradient = gradient
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.reg_l2 = reg_l2
        self.on_epoch = on_epoch
        self.state: ApproxState | None = None

    def fit(self, X: Any, y: np.ndarray, state: ApproxState | None = None) -> "ApproxLogisticModel":
        """Обучение; state позволяет продолжить с сохранённой эпохи."""
        y = np.asarray(y, dtype=float)
        self.state = state if state is not None else ApproxState.start(y, X.shape[1])
        run_epochs(
            self.state,
            X,
            y,
            self.gradient or plain_gradient,
            self.epochs,
            self.learning_rate,
            self.reg_l2,
            self.on_epoch,
        )
        return self

    @property
    def weights(self) -> np.ndarray:
        return self._trained().weights

    @property
    def bias(self) -> float:
        return self._trained().bias

    def predict_proba(self, X: Any) -> np.ndarray:
        state = self._trained()
        return sigmoid(np.asarray(X @ state.weights).ravel() + state.bias)

    def _trained(self) -> ApproxState:
        if self.state is None:
            raise RuntimeError(f"Модель «{self.name}» не обучена: сначала вызовите fit")
        return self.state


class BankOnlyModel(ApproxLogisticModel):
    """База: только признаки банка."""

    name = "bank_only"


class CeilingModel(ApproxLogisticModel):
    """Потолок: признаки банка и все признаки телекома в открытую (лаборатория)."""

    name = "ceiling"


class WoEScorecard(ScoringModel):
    """Модель из агрегатов: скор банка + поправка по WoE бинов телекома.

    ЗАГЛУШКА: в текущих экспериментах не используется. Оставлена в интерфейсе,
    чтобы её можно было реализовать позже, не меняя демо и тесты.

    woe_tables: признак -> таблица из smpc.metrics.woe_iv (колонки bin, WoE).
    """

    name = "woe_scorecard"

    def __init__(self, base_model: ScoringModel, woe_tables: dict[str, pd.DataFrame]) -> None:
        self.base_model = base_model
        self.woe_tables = woe_tables

    def fit(self, X: Any, y: np.ndarray) -> "WoEScorecard":
        raise NotImplementedError("ML-часть: скоркарта на WoE из агрегатов PIS (не реализовано)")

    def predict_proba(self, X: Any) -> np.ndarray:
        raise NotImplementedError("ML-часть: скоркарта на WoE из агрегатов PIS (не реализовано)")


def compare_models(evaluations: Mapping[str, tuple[ScoringModel, Any]], y_test: np.ndarray) -> pd.DataFrame:
    """Таблица «модель → метрики» на отложенной выборке.

    evaluations: название -> (обученная модель, её тестовая матрица). Матрицы
    у разных моделей могут отличаться по числу признаков.
    """
    from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

    rows = []
    for label, (model, X_test) in evaluations.items():
        proba = model.predict_proba(X_test)
        rows.append({
            "Модель": label,
            "Признаков": X_test.shape[1],
            "ROC-AUC": float(roc_auc_score(y_test, proba)),
            "PR-AUC": float(average_precision_score(y_test, proba)),
            "Log-loss": float(log_loss(y_test, proba, labels=[0, 1])),
        })
    return pd.DataFrame(rows).set_index("Модель")
