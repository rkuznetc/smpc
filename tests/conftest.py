"""Маленькие наборы с ЧАСТИЧНЫМ пересечением: у каждой стороны есть свои клиенты.

Банк:    100001..100010
Телеком: 100006..100015
Общие:   100006..100010 (5 клиентов)
"""

import numpy as np
import pandas as pd
import pytest

from smpc.data import prepare_bank, prepare_telecom
from smpc.parties import Bank, Telecom

BANK_IDS = list(range(100001, 100011))
TELECOM_IDS = list(range(100006, 100016))
COMMON = {str(i) for i in range(100006, 100011)}


def bank_frame() -> pd.DataFrame:
    return pd.DataFrame({"SK_ID_CURR": BANK_IDS, "TARGET": [0, 1, 0, 0, 1, 0, 1, 0, 0, 1]})


def telecom_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "SK_ID_CURR": TELECOM_IDS,
        "DAYS_LAST_PHONE_CHANGE": [-100, -500, -2000, np.nan, -30, -400, 5, -1200, -10, -800],
        "DEF_30_CNT_SOCIAL_CIRCLE": [0, 1, 2, np.nan, 0, 3, -1, 0, 1, 0],
    })


@pytest.fixture
def bank() -> Bank:
    return Bank(prepare_bank(bank_frame()))


@pytest.fixture
def telecom() -> Telecom:
    return Telecom(prepare_telecom(telecom_frame(), ["DAYS_LAST_PHONE_CHANGE", "DEF_30_CNT_SOCIAL_CIRCLE"]))
