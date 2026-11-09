"""CKKS: зашифрованный градиент приближённой логистической регрессии.

Что делает CKKSGradient (логика ноутбуков 04 и 05):
  для каждого пакета строк шифрует признаки по столбцам, вычисляет
  z = Xw + b, остаток 0.25·z + 0.5 − y и его произведения с признаками,
  суммирует по пакетам и передаёт агрегаты на расшифрование.

ВАЖНО о безопасности:
  DemoAggregateDecryptor — учебный «сервис расшифрования»: прибавляет к
  агрегату случайную маску, расшифровывает и снимает маску в одном процессе.
  Это НЕ безопасный протокол: расшифровывающая сторона видит агрегаты.
  Класс вынесен отдельно, чтобы его можно было заменить настоящим протоколом
  с распределённым ключом, не меняя CKKSGradient.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class CKKSParams:
    poly_modulus_degree: int = 8192
    coeff_mod_bit_sizes: tuple[int, ...] = (50, 30, 30, 30, 50)
    global_scale: float = 2.0**30


def require_tenseal():
    try:
        import tenseal as ts
    except ImportError as exc:  # pragma: no cover - зависит от окружения
        raise ImportError("Для CKKS нужен tenseal: pip install -e '.[he]'") from exc
    return ts


def make_context_pair(params: CKKSParams = CKKSParams()) -> tuple[Any, Any]:
    """(приватный контекст с секретным ключом, публичный контекст без него)."""
    ts = require_tenseal()
    private = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=params.poly_modulus_degree,
        coeff_mod_bit_sizes=list(params.coeff_mod_bit_sizes),
    )
    private.global_scale = params.global_scale
    private.generate_galois_keys()
    private.generate_relin_keys()
    public = ts.context_from(private.serialize(save_secret_key=False))
    public.global_scale = params.global_scale
    if public.has_secret_key():
        raise RuntimeError("Публичный контекст не должен содержать секретный ключ")
    return private, public


class DemoAggregateDecryptor:
    """Учебное расшифрование агрегатов с маской. НЕ безопасный протокол (см. модуль)."""

    def __init__(self, private_context: Any, mask_range: float = 16.0) -> None:
        self._private = private_context
        self._mask_range = mask_range
        self._rng = secrets.SystemRandom()

    def __call__(self, encrypted_scalar: Any) -> float:
        ts = require_tenseal()
        mask = self._rng.uniform(-self._mask_range, self._mask_range)
        payload = (encrypted_scalar + mask).serialize()
        value = float(ts.ckks_vector_from(self._private, payload).decrypt()[0])
        return value - mask


class CKKSGradient:
    """Зашифрованный приближённый градиент. Совместим с run_epochs (см. ml/approx.py)."""

    def __init__(
        self,
        public_context: Any,
        decrypt_aggregate: Callable[[Any], float],
        batch_size: int = 1024,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size должен быть положительным")
        self._context = public_context
        self._decrypt = decrypt_aggregate
        self.batch_size = batch_size

    def __call__(self, w: np.ndarray, b: float, X: Any, y: np.ndarray) -> tuple[np.ndarray, float]:
        ts = require_tenseal()
        X = sp.csr_matrix(X)
        n, p = X.shape
        if p < 1:
            raise ValueError("Нужен хотя бы один признак")
        y = np.asarray(y, dtype=float)
        totals: list[Any] = [None] * (p + 1)

        for offset in range(0, n, self.batch_size):
            block = X[offset:offset + self.batch_size].toarray()
            # В реальном протоколе каждая сторона шифрует только свою часть блока.
            columns = [ts.ckks_vector(self._context, block[:, j].tolist()) for j in range(p)]
            encrypted_y = ts.ckks_vector(self._context, y[offset:offset + self.batch_size].tolist())

            z = columns[0] * float(w[0])
            for j in range(1, p):
                z += columns[j] * float(w[j])
            z += float(b)
            error = z * 0.25 + 0.5 - encrypted_y

            parts = [(error * column).sum() for column in columns] + [error.sum()]
            for j, part in enumerate(parts):
                totals[j] = part if totals[j] is None else totals[j] + part

        sums = np.array([self._decrypt(total) for total in totals]) / n
        return sums[:-1], float(sums[-1])
