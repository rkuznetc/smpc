"""Общий интерфейс протоколов.

Любой протокол (PSI, PSI-CA, PIS и будущие) принимает стороны и канал
и возвращает ProtocolResult одного вида. Поэтому демо сводит все
протоколы в одну таблицу, а новый метод добавляется одним классом.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from smpc.channel import Channel
from smpc.parties import Bank, Telecom


@dataclass
class ProtocolResult:
    name: str
    bank_output: Any  # что получил банк
    telecom_output: Any  # что получил телеком
    seconds: float
    bytes_sent: int
    leakage: dict[str, str]  # декларация: кто что узнаёт
    details: dict[str, Any] = field(default_factory=dict)


class Protocol(ABC):
    """Базовый класс протокола."""

    name: ClassVar[str] = "protocol"
    #: Кто что узнаёт (заполняется в подклассе, попадает в отчёт демо).
    leakage: ClassVar[dict[str, str]] = {}

    def run(self, bank: Bank, telecom: Telecom, channel: Channel | None = None) -> ProtocolResult:
        channel = channel if channel is not None else Channel()
        start = time.perf_counter()
        bank_output, telecom_output, details = self._execute(bank, telecom, channel)
        seconds = time.perf_counter() - start
        return ProtocolResult(
            name=self.name,
            bank_output=bank_output,
            telecom_output=telecom_output,
            seconds=seconds,
            bytes_sent=channel.bytes_total,
            leakage=dict(self.leakage),
            details=details,
        )

    @abstractmethod
    def _execute(self, bank: Bank, telecom: Telecom, channel: Channel) -> tuple[Any, Any, dict[str, Any]]:
        """Выполняет протокол; возвращает (выход банка, выход телекома, детали)."""
