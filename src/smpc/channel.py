"""Канал между сторонами: передаёт байты и считает трафик.

Стороны в MVP работают в одном процессе, но все сообщения протоколов
проходят через Channel в сериализованном виде. Это даёт:
  * явную границу «что ушло от одной стороны к другой»;
  * метрику трафика (байты) для сравнения протоколов.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class Message:
    sender: str
    receiver: str
    label: str
    size: int


@dataclass
class Channel:
    log: list[Message] = field(default_factory=list)

    def send(self, sender: str, receiver: str, label: str, payload: bytes) -> bytes:
        """Записывает сообщение в журнал и возвращает его получателю без изменений."""
        if not isinstance(payload, (bytes, bytearray)):
            raise TypeError("По каналу передаются только байты")
        self.log.append(Message(sender, receiver, label, len(payload)))
        return bytes(payload)

    @property
    def bytes_total(self) -> int:
        return sum(m.size for m in self.log)

    def summary(self) -> pd.DataFrame:
        """Журнал сообщений: кто, кому, что и сколько байт."""
        return pd.DataFrame([m.__dict__ for m in self.log], columns=["sender", "receiver", "label", "size"])
