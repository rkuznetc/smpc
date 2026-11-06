"""Эталон: открытый JOIN. ТОЛЬКО для лабораторной проверки корректности.

При реальных двух организациях такой расчёт недопустим: он требует
данных обеих сторон в одном месте (как отмечено в ноутбуке 02).
"""

from __future__ import annotations

import pandas as pd

from smpc.parties import Bank, Telecom


def intersection(bank: Bank, telecom: Telecom) -> set[str]:
    return set(bank.ids) & set(telecom.ids)


def intersection_target_sum(bank: Bank, telecom: Telecom) -> int:
    common = intersection(bank, telecom)
    targets = bank.targets
    return int(targets[targets.index.isin(common)].sum())


def bin_stats(bank: Bank, telecom: Telecom, feature: str) -> pd.DataFrame:
    """N и сумма TARGET по бинам признака телекома среди общих клиентов."""
    bins = telecom.bins(feature).rename_axis("ID").reset_index()
    joined = bank.data[["ID", "TARGET"]].merge(bins, on="ID", how="inner", validate="one_to_one")
    out = (
        joined.groupby("bin", sort=True)
        .agg(N=("TARGET", "size"), defaults=("TARGET", "sum"))
        .reset_index()
    )
    out["N"] = out["N"].astype(int)
    out["defaults"] = out["defaults"].astype(int)
    return out
