"""Контрольные точки обучения: сохранение после каждой эпохи и возобновление.

Отпечаток (fingerprint) связывает файл с конкретной выборкой, признаками и
параметрами. Если они изменились, загрузка падает с понятной ошибкой: старый
файл не перезаписывается и не используется молча.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from smpc.ml.approx import ApproxState

METHOD = "ckks_linear_approx_sigmoid_v1"


class CheckpointMismatch(ValueError):
    """Контрольная точка относится к другим данным или параметрам."""


def fingerprint(X: Any, y: np.ndarray, names: list[str], config: dict[str, Any]) -> str:
    """SHA-256 по конфигурации, названиям признаков, форме и содержимому данных.

    Число эпох в отпечаток не входит: обучение можно продолжить на большее число эпох.
    """
    X = sp.csr_matrix(X)
    payload = {"names": list(names), "shape": list(X.shape), "method": METHOD, **config}
    digest = hashlib.sha256()
    digest.update(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    for array in (X.indptr, X.indices, X.data, np.asarray(y, dtype=np.float64)):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def save_state(path: str | Path, state: ApproxState, fp: str, names: list[str]) -> None:
    """Атомарная запись: сначала временный файл, затем os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        weights=state.weights,
        bias=np.array(state.bias),
        epoch=np.array(state.epoch),
        fingerprint=np.array(fp),
        history_json=np.array(json.dumps(state.history)),
        feature_names=np.array(names),
    )
    os.replace(temporary, path)


def load_state(path: str | Path, fp: str) -> ApproxState | None:
    """Состояние из файла или None, если файла нет. Несовпадение отпечатка — ошибка."""
    path = Path(path)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as saved:
        if str(saved["fingerprint"].item()) != fp:
            raise CheckpointMismatch(
                f"Контрольная точка {path} создана для другой выборки, признаков или параметров. "
                "Укажите другую папку --checkpoint-dir или удалите файл вручную."
            )
        return ApproxState(
            weights=saved["weights"].astype(float),
            bias=float(saved["bias"].item()),
            epoch=int(saved["epoch"].item()),
            history=json.loads(str(saved["history_json"].item())),
        )
