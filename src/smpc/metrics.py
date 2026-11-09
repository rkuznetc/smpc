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


def best_f1_threshold(y_true, probabilities) -> float:
    """Порог, максимизирующий F1. Подбирается только на validation, не на test."""
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def classification_row(y_true, probabilities, threshold: float) -> dict[str, float]:
    """Метрики качества вероятностей и бинарного решения при заданном пороге."""
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    predicted = (np.asarray(probabilities) >= threshold).astype(int)
    return {
        "ROC-AUC": float(roc_auc_score(y_true, probabilities)),
        "PR-AUC": float(average_precision_score(y_true, probabilities)),
        "Log-loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "Brier": float(brier_score_loss(y_true, probabilities)),
        "Precision": float(precision_score(y_true, predicted, zero_division=0)),
        "Recall": float(recall_score(y_true, predicted, zero_division=0)),
        "F1": float(f1_score(y_true, predicted, zero_division=0)),
        "Balanced accuracy": float(balanced_accuracy_score(y_true, predicted)),
    }
