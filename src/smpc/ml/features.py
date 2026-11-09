"""Подготовка признаков для ML-эксперимента (логика ноутбуков 04 и 05).

* Служебные колонки (ID, SK_ID_*, TARGET) исключаются.
* Числовые признаки: медианная импутация, стандартизация, ограничение ±5.
* Категориальные: One-Hot (неизвестные категории игнорируются).
* Препроцессоры обучаются только на train. L1-отбор — только на train.
Открытый L1-отбор — исследовательский; в разделённой системе он потребует
конфиденциальной замены.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CLIP_VALUE = 5.0
MISSING_CATEGORY = "__MISSING__"


def is_id_like(name: str) -> bool:
    upper = name.upper()
    return (
        upper in {"ID", "SK_ID_CURR", "TARGET", "INDEX", "ROW_ID"}
        or upper.endswith("_ID")
        or upper.startswith("SK_ID_")
    )


def select_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Все колонки, которые могут быть признаками (без ID и метки)."""
    return [c for c in frame.columns if not is_id_like(c)]


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Числа — в числовой тип (inf → NaN), остальное — строки с отдельной категорией пропусков."""
    frame = frame.copy()
    for c in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[c]):
            frame[c] = pd.to_numeric(frame[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        else:
            frame[c] = frame[c].astype("string").fillna(MISSING_CATEGORY).astype(str)
    return frame


def make_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in frame.columns if c not in numeric]
    steps = []
    if numeric:
        steps.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
            ]),
            numeric,
        ))
    if categorical:
        steps.append((
            "categorical",
            Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32))]),
            categorical,
        ))
    if not steps:
        raise ValueError("Нет признаков после отбора колонок")
    return ColumnTransformer(steps, sparse_threshold=1.0)


def clip_sparse(matrix: sp.csr_matrix, limit: float = CLIP_VALUE) -> sp.csr_matrix:
    """Ограничивает ненулевые значения ±limit (нули остаются нулями)."""
    matrix.data = np.clip(matrix.data, -limit, limit)
    matrix.eliminate_zeros()
    return matrix


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def make_splits(
    y: np.ndarray, seed: int = 42, test_size: float = 0.2, val_size: float = 0.2
) -> SplitIndices:
    """Стратифицированные train / validation / test (validation — часть train)."""
    indices = np.arange(len(y))
    train_val, test = train_test_split(indices, test_size=test_size, random_state=seed, stratify=y)
    train, val = train_test_split(train_val, test_size=val_size, random_state=seed, stratify=y[train_val])
    return SplitIndices(train=train, val=val, test=test)


@dataclass
class FeatureBlock:
    """Матрицы одной стороны (банка или телекома) после препроцессинга."""

    train: sp.csr_matrix
    val: sp.csr_matrix
    test: sp.csr_matrix
    names: list[str]


def build_block(frame: pd.DataFrame, prefix: str, splits: SplitIndices) -> FeatureBlock:
    """Препроцессинг одной стороны; fit только на train-строках."""
    raw = clean_frame(frame[select_feature_columns(frame)])
    transformer = make_preprocessor(raw)
    train = transformer.fit_transform(raw.iloc[splits.train])
    val = transformer.transform(raw.iloc[splits.val])
    test = transformer.transform(raw.iloc[splits.test])
    matrices = [clip_sparse(sp.csr_matrix(m, dtype=np.float32)) for m in (train, val, test)]
    names = [prefix + n for n in transformer.get_feature_names_out()]
    return FeatureBlock(train=matrices[0], val=matrices[1], test=matrices[2], names=names)


def l1_logistic(C: float, seed: int) -> LogisticRegression:
    """L1-логистическая регрессия; параметр penalty устарел в sklearn ≥ 1.8 (заменён на l1_ratio)."""
    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    kwargs: dict[str, Any] = {"solver": "liblinear", "C": C, "max_iter": 500, "random_state": seed}
    if (major, minor) >= (1, 8):
        kwargs["l1_ratio"] = 1.0
    else:
        kwargs["penalty"] = "l1"
    return LogisticRegression(**kwargs)


def l1_select(
    X_train: sp.csr_matrix,
    y_train: np.ndarray,
    names: list[str],
    max_features: int | None,
    C: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """L1-логистическая регрессия на train: индексы признаков с ненулевыми коэффициентами.

    Не более max_features признаков (по убыванию |коэффициента|, при равенстве — по индексу).
    Возвращает (отсортированные индексы, |коэффициенты| по всем признакам).
    """
    if len(names) != X_train.shape[1]:
        raise ValueError("Число названий признаков не совпадает с числом столбцов")
    selector = l1_logistic(C, seed)
    selector.fit(X_train, y_train.astype(int))
    importance = np.abs(selector.coef_[0])
    nonzero = np.flatnonzero(importance > 1e-10)
    if nonzero.size == 0:
        raise ValueError("L1 обнулил все признаки: увеличьте C (например, до 1.0)")
    ranked = sorted(nonzero, key=lambda j: (-importance[j], j))
    if max_features is not None:
        ranked = ranked[:max_features]
    return np.array(sorted(ranked), dtype=int), importance
