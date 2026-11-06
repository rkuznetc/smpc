"""Стороны протокола.

Каждая сторона держит только свои данные. Протоколы обращаются к данным
стороны через её методы и обмениваются сообщениями через Channel —
так в коде видно, что пересекает границу между банком и телекомом.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from smpc.data import assign_bins


@dataclass
class Bank:
    """Банк: ID клиентов и TARGET (дефолт 0/1). Получатель результатов."""

    data: pd.DataFrame  # колонки: ID, TARGET

    name = "bank"

    @property
    def ids(self) -> list[str]:
        """Отсортированный список ID (как bank_new в ноутбуках)."""
        return sorted(self.data["ID"].tolist())

    @property
    def targets(self) -> pd.Series:
        """TARGET, индексированный по ID."""
        return self.data.set_index("ID")["TARGET"]

    def __len__(self) -> int:
        return len(self.data)


@dataclass
class Telecom:
    """Телеком: ID абонентов и признаки абонентов."""

    data: pd.DataFrame  # колонки: ID и признаки

    name = "telecom"

    @property
    def ids(self) -> list[str]:
        """Отсортированный список ID (как telecom_new в ноутбуках)."""
        return sorted(self.data["ID"].tolist())

    def bins(self, feature: str) -> pd.Series:
        """Бин выбранного признака для каждого абонента, индекс — ID."""
        if feature not in self.data.columns:
            raise ValueError(f"У телекома нет признака {feature}")
        return pd.Series(assign_bins(self.data[feature], feature).to_numpy(), index=self.data["ID"], name="bin")

    def __len__(self) -> int:
        return len(self.data)
