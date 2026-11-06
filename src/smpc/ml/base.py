"""Скелет под будущий ML. Экспериментов пока нет — только интерфейсы.

Планируемое сравнение на отложенной выборке общих клиентов:
    BankOnlyModel  — база: только признаки банка;
    CeilingModel   — потолок: признаки банка + 5 признаков телекома
                     в открытую (возможно только в лаборатории);
    WoEScorecard   — модель из агрегатов PIS: поправка к скору банка
                     по WoE бинов телекома.
Метрика: ROC-AUC; итог — доля выигрыша потолка, которую сохранила
модель из агрегатов.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class ScoringModel(ABC):
    """Модель скоринга: вероятность дефолта по признакам клиента."""

    name: str = "model"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ScoringModel":
        """Обучение. Возвращает self."""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Вероятность дефолта для каждой строки X."""


class BankOnlyModel(ScoringModel):
    """База: только признаки банка."""

    name = "bank_only"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BankOnlyModel":
        raise NotImplementedError("ML-часть: модель на признаках банка")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("ML-часть: модель на признаках банка")


class CeilingModel(ScoringModel):
    """Потолок: признаки банка + признаки телекома, соединённые в открытую."""

    name = "ceiling"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CeilingModel":
        raise NotImplementedError("ML-часть: модель на признаках банка и телекома")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("ML-часть: модель на признаках банка и телекома")


class WoEScorecard(ScoringModel):
    """Модель из агрегатов: скор банка + поправка по WoE бинов телекома.

    woe_tables: признак -> таблица из smpc.metrics.woe_iv (колонки bin, WoE).
    """

    name = "woe_scorecard"

    def __init__(self, base_model: ScoringModel, woe_tables: dict[str, pd.DataFrame]) -> None:
        self.base_model = base_model
        self.woe_tables = woe_tables

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoEScorecard":
        raise NotImplementedError("ML-часть: скоркарта на WoE из агрегатов PIS")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("ML-часть: скоркарта на WoE из агрегатов PIS")


def compare_models(models: list[ScoringModel], X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Таблица «модель → ROC-AUC» на отложенной выборке."""
    raise NotImplementedError("ML-часть: сравнение моделей по ROC-AUC")
