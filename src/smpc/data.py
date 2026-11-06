"""Загрузка данных сторон и построение бинов.

Логика повторяет ноутбуки 02_psi_psica_full и 03_pis_groups_prototype:
ID = SK_ID_CURR (int64 -> str), TARGET бинарный, дубликаты удаляются,
конфликтующие значения для одного ID — ошибка.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BANK_FILE = "bank_psi.parquet"
TELECOM_FILE = "telecom_train.parquet"
ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"

TELECOM_FEATURES = [
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "OBS_60_CNT_SOCIAL_CIRCLE",
    "DEF_60_CNT_SOCIAL_CIRCLE",
    "DAYS_LAST_PHONE_CHANGE",
]

MISSING_BIN = "Пропуск"


def normalize_ids(series: pd.Series) -> pd.Series:
    """SK_ID_CURR -> строка: int64, затем str (как в ноутбуках)."""
    return series.astype("int64").astype(str)


def prepare_bank(frame: pd.DataFrame) -> pd.DataFrame:
    """Таблица банка -> колонки ID (str) и TARGET (0/1)."""
    df = frame[[ID_COLUMN, TARGET_COLUMN]].dropna(subset=[ID_COLUMN, TARGET_COLUMN]).copy()
    if df.groupby(ID_COLUMN)[TARGET_COLUMN].nunique().gt(1).any():
        raise ValueError("Обнаружены конфликтующие TARGET для одного SK_ID_CURR")
    df = df.drop_duplicates(ID_COLUMN)
    if not df[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("TARGET должен быть бинарным (0/1)")
    return pd.DataFrame(
        {"ID": normalize_ids(df[ID_COLUMN]).to_numpy(), "TARGET": df[TARGET_COLUMN].astype("int64").to_numpy()}
    )


def prepare_telecom(frame: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """Таблица телекома -> колонка ID (str) и выбранные признаки."""
    features = list(features or [])
    missing = [f for f in features if f not in frame.columns]
    if missing:
        raise ValueError(f"В таблице телекома нет признаков: {missing}")
    df = frame[[ID_COLUMN, *features]].dropna(subset=[ID_COLUMN]).copy()
    for feature in features:
        if df.groupby(ID_COLUMN)[feature].nunique(dropna=True).gt(1).any():
            raise ValueError(f"У одного ID разные значения признака {feature}: нужна агрегация")
    df = df.drop_duplicates(ID_COLUMN)
    out = df[features].reset_index(drop=True)
    out.insert(0, "ID", normalize_ids(df[ID_COLUMN]).to_numpy())
    return out


def load_bank(path: str | Path) -> pd.DataFrame:
    """Читает только SK_ID_CURR и TARGET из parquet банка."""
    return prepare_bank(pd.read_parquet(path, columns=[ID_COLUMN, TARGET_COLUMN]))


def load_telecom(path: str | Path, features: list[str] | None = None) -> pd.DataFrame:
    """Читает SK_ID_CURR и выбранные признаки из parquet телекома."""
    features = list(features or [])
    return prepare_telecom(pd.read_parquet(path, columns=[ID_COLUMN, *features]), features)


def assign_bins(series: pd.Series, feature: str) -> pd.Series:
    """Бины признака телекома (как в ноутбуке 03).

    DAYS_LAST_PHONE_CHANGE: давность смены номера в днях (значение со знаком минус)
        -> «0–1 год», «1–3 года», «более 3 лет».
    Остальные признаки (*_CNT_SOCIAL_CIRCLE) -> «0», «1», «2+».
    Пропуски и отрицательные/аномальные значения -> «Пропуск».
    """
    v = pd.to_numeric(series, errors="coerce")
    if feature == "DAYS_LAST_PHONE_CHANGE":
        age = -v
        bins = pd.cut(
            age.where(age >= 0),
            [-0.001, 365, 1095, np.inf],
            labels=["0–1 год", "1–3 года", "более 3 лет"],
        )
    else:
        bins = pd.cut(v.where(v >= 0), [-0.001, 0, 1, np.inf], labels=["0", "1", "2+"])
    return bins.astype("object").fillna(MISSING_BIN)


def subsample(frame: pd.DataFrame, n: int | None, seed: int = 0) -> pd.DataFrame:
    """Случайная подвыборка n строк (None — без подвыборки, как в ноутбуках)."""
    if n is None or n >= len(frame):
        return frame
    return frame.sample(n=n, random_state=seed).reset_index(drop=True)
