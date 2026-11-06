"""PIS по бинам признака — ПРОТОТИП (уровень 2), логика ноутбука 03.

Что делает:
  1. Обе стороны строят HMAC-псевдонимы ID на общем случайном ключе.
  2. Банк передаёт свои псевдонимы телекому (сопоставляет телеком).
  3. В доверенной демо-среде псевдонимы сопоставляются в открытую и
     считаются агрегаты по бинам: N (общие клиенты) и defaults (сумма TARGET).
  4. Агрегаты шифруются открытым ключом Paillier банка, передаются банку
     и расшифровываются его секретным ключом.

Это НЕ криптографически безопасный двухсторонний Private Intersection-Sum:
  * сопоставление и агрегация выполняются в открытую (шаг 3);
  * телеком видит, какие его абоненты совпали с банком;
  * при общем HMAC-ключе каждая сторона может перебором восстановить ID другой;
  * Paillier защищает только передачу уже посчитанных агрегатов.
Агрегаты, WoE и IV при этом корректны — это контрольная реализация того,
что должен выдавать настоящий PIS (см. protocols/future.py).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from typing import Any

import numpy as np
import pandas as pd
from phe import paillier

from smpc.channel import Channel
from smpc.parties import Bank, Telecom
from smpc.protocols.base import Protocol

DIGEST_SIZE = 32  # HMAC-SHA256


def hmac_token(key: bytes, id_value: str) -> bytes:
    """HMAC-псевдоним ID (канонизация str(int(...)), как в ноутбуке)."""
    canonical = str(int(id_value)).encode("utf-8")
    return hmac.new(key, canonical, hashlib.sha256).digest()


def trusted_bin_aggregates(bank_tokens: pd.DataFrame, telecom_tokens: pd.DataFrame) -> pd.DataFrame:
    """Доверенная демо-среда: открытое сопоставление и агрегация по бинам.

    В настоящем PIS этот шаг должен выполняться под защитой.
    """
    joined = bank_tokens.merge(telecom_tokens, on="token", how="inner", validate="one_to_one")
    counts = (
        joined.groupby("bin", sort=True)
        .agg(N=("TARGET", "size"), defaults=("TARGET", "sum"))
        .reset_index()
    )
    counts["N"] = counts["N"].astype(int)
    counts["defaults"] = counts["defaults"].astype(int)
    return counts


class PISPrototype(Protocol):
    """Прототип PIS: банк получает N и сумму TARGET по бинам признака телекома."""

    name = "PIS (прототип)"
    leakage = {
        "bank": "N и сумма TARGET по бинам признака телекома",
        "telecom": "какие его абоненты совпали с банком; по общему HMAC-ключу может перебрать ID банка",
        "demo_env": "все данные в открытую (сопоставление и агрегация до шифрования)",
    }

    def __init__(self, feature: str, key_bits: int = 2048) -> None:
        self.feature = feature
        self.key_bits = key_bits

    def _execute(self, bank: Bank, telecom: Telecom, channel: Channel) -> tuple[Any, Any, dict[str, Any]]:
        # 1. Общий HMAC-ключ, о котором договорились стороны (как в ноутбуке).
        shared_key = secrets.token_bytes(32)
        bank_tokens = pd.DataFrame(
            {"token": [hmac_token(shared_key, i) for i in bank.data["ID"]], "TARGET": bank.data["TARGET"].to_numpy()}
        )
        bins = telecom.bins(self.feature)
        telecom_tokens = pd.DataFrame(
            {"token": [hmac_token(shared_key, i) for i in bins.index], "bin": bins.to_numpy()}
        )

        # 2. Банк -> телеком: псевдонимы своих клиентов.
        sent = channel.send("bank", "telecom", "hmac_tokens", b"".join(bank_tokens["token"]))
        n_tokens_received = len(sent) // DIGEST_SIZE

        # 3. Доверенная демо-среда: открытое сопоставление и агрегаты.
        counts = trusted_bin_aggregates(bank_tokens, telecom_tokens)
        n_matched = int(counts["N"].sum())

        # 4. Банк генерирует ключи Paillier, публичный ключ уходит в демо-среду.
        public_key, private_key = paillier.generate_paillier_keypair(n_length=self.key_bits)
        n_bytes = channel.send("bank", "demo_env", "paillier_public_key", _int_to_bytes(public_key.n))
        env_public_key = paillier.PaillierPublicKey(int.from_bytes(n_bytes, "big"))

        encrypted = [
            (row.bin, env_public_key.encrypt(int(row.N)), env_public_key.encrypt(int(row.defaults)))
            for row in counts.itertuples(index=False)
        ]
        payload = json.dumps(
            [{"bin": b, "N": hex(n.ciphertext()), "defaults": hex(d.ciphertext())} for b, n, d in encrypted]
        ).encode("utf-8")
        received = json.loads(channel.send("demo_env", "bank", "encrypted_aggregates", payload))

        # Банк расшифровывает только суммы по группам.
        rows, enc_defaults = [], []
        for item in received:
            enc_n = paillier.EncryptedNumber(public_key, int(item["N"], 16))
            enc_d = paillier.EncryptedNumber(public_key, int(item["defaults"], 16))
            enc_defaults.append(enc_d)
            rows.append((item["bin"], private_key.decrypt(enc_n), private_key.decrypt(enc_d)))
        result = pd.DataFrame(rows, columns=["bin", "N", "defaults"])
        result["default_rate"] = result["defaults"] / result["N"].replace(0, np.nan)

        # Проверка сложения под шифром (как в ноутбуке).
        total = sum(enc_defaults, public_key.encrypt(0))
        homomorphic_ok = private_key.decrypt(total) == int(result["defaults"].sum())

        telecom_output = f"{n_tokens_received} псевдонимов банка, из них совпало {n_matched}"
        details = {"feature": self.feature, "homomorphic_check": homomorphic_ok, "n_common": n_matched}
        return result, telecom_output, details


def _int_to_bytes(value: int) -> bytes:
    return value.to_bytes((value.bit_length() + 7) // 8, "big")
