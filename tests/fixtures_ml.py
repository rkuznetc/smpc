"""Синтетические таблицы для ML-тестов: частичное пересечение, сигнал в признаках."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_frames(n: int = 200, drop: int = 20, extra: int = 10, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Банк: n клиентов. Телеком: без последних `drop` клиентов банка + `extra` чужих.

    Общих клиентов: n - drop. Дефолт зависит от EXT_SCORE, поэтому сигнал есть.
    """
    rng = np.random.default_rng(seed)
    ids = np.arange(100000, 100000 + n)
    score = rng.normal(size=n)
    target = (rng.random(n) < 1 / (1 + np.exp(-(0.9 * score - 1.2)))).astype(int)
    amount = rng.normal(loc=5e5, scale=1e5, size=n)
    amount[rng.random(n) < 0.05] = np.nan
    bank = pd.DataFrame({
        "SK_ID_CURR": ids,
        "SK_ID_BUREAU": rng.integers(1, 10**6, n),
        "TARGET": target,
        "AMT_CREDIT": amount,
        "DAYS_BIRTH": -rng.integers(7000, 25000, n),
        "CODE_GENDER": rng.choice(["F", "M"], n),
        "EXT_SCORE": score + rng.normal(scale=0.5, size=n),
    })
    tel_ids = np.concatenate([ids[: n - drop], np.arange(200000, 200000 + extra)])
    m = len(tel_ids)
    telecom = pd.DataFrame({
        "SK_ID_CURR": tel_ids,
        "DAYS_LAST_PHONE_CHANGE": -rng.integers(10, 3000, m).astype(float),
        "DEF_30_CNT_SOCIAL_CIRCLE": rng.integers(0, 4, m).astype(float),
        "OBS_30_CNT_SOCIAL_CIRCLE": rng.integers(0, 6, m).astype(float),
        "OBS_60_CNT_SOCIAL_CIRCLE": rng.integers(0, 6, m).astype(float),
        "DEF_60_CNT_SOCIAL_CIRCLE": rng.integers(0, 4, m).astype(float),
    })
    return bank, telecom


def write_parquets(directory: Path, **kwargs) -> Path:
    bank, telecom = make_frames(**kwargs)
    directory.mkdir(parents=True, exist_ok=True)
    bank.to_parquet(directory / "bank_psi.parquet")
    telecom.to_parquet(directory / "telecom_train.parquet")
    return directory
