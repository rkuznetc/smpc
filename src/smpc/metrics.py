"""WoE и IV по агрегатам бинов (формулы ноутбука 03).

WoE = ln(p_nondefault / p_default),
IV  = Σ (p_nondefault − p_default) · WoE,
где p — доля бина среди всех недефолтов / дефолтов, со сглаживанием
smoothing в каждом бине (по умолчанию 0.5, как в ноутбуке).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def woe_iv(table: pd.DataFrame, smoothing: float = 0.5) -> tuple[pd.DataFrame, float]:
    """table: колонки bin, N, defaults. Возвращает (таблица с WoE, IV)."""
    result = table[["bin", "N", "defaults"]].copy()
    result["nondefaults"] = result["N"] - result["defaults"]
    num_bins = len(result)
    pos_total = result["defaults"].sum()
    neg_total = result["nondefaults"].sum()
    result["p_default"] = (result["defaults"] + smoothing) / (pos_total + smoothing * num_bins)
    result["p_nondefault"] = (result["nondefaults"] + smoothing) / (neg_total + smoothing * num_bins)
    with np.errstate(divide="ignore"):
        result["WoE"] = np.log(result["p_nondefault"] / result["p_default"])
    result["IV_contrib"] = (result["p_nondefault"] - result["p_default"]) * result["WoE"]
    return result, float(result["IV_contrib"].sum())
