"""Заглушки протоколов, которые могут появиться позже.

Интерфейс тот же (Protocol.run -> ProtocolResult), поэтому после
реализации они сразу встанут в демо и тесты. Пока вызывают
NotImplementedError.
"""

from __future__ import annotations

from typing import Any

from smpc.channel import Channel
from smpc.parties import Bank, Telecom
from smpc.protocols.base import Protocol


class PrivateIntersectionSum(Protocol):
    """Настоящий двухсторонний PIS (модель semi-honest).

    Ожидаемый результат тот же, что у PISPrototype (N и сумма TARGET по бинам),
    но без открытого сопоставления: ни банк, ни телеком не узнают,
    какие именно клиенты совпали.
    """

    name = "PIS"
    leakage = {
        "bank": "N и сумма TARGET по бинам",
        "telecom": "размер пересечения",
    }

    def __init__(self, feature: str) -> None:
        self.feature = feature

    def _execute(self, bank: Bank, telecom: Telecom, channel: Channel) -> tuple[Any, Any, dict[str, Any]]:
        raise NotImplementedError("Настоящий PIS пока не реализован; см. PISPrototype")


class PS3I(Protocol):
    """Пересечение, где значения выходят долями секрета для дальнейших вычислений (MPC)."""

    name = "PS3I"
    leakage = {
        "bank": "доли секрета значений по общим клиентам",
        "telecom": "доли секрета значений по общим клиентам",
    }

    def _execute(self, bank: Bank, telecom: Telecom, channel: Channel) -> tuple[Any, Any, dict[str, Any]]:
        raise NotImplementedError("PS3I пока не реализован")
