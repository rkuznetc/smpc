import math

import numpy as np
import pandas as pd
import pytest

from smpc.metrics import woe_iv

# Пример: 1000 общих клиентов, 900 без дефолта, 100 с дефолтом.
TABLE = pd.DataFrame({"bin": ["0", "1", "2+"], "N": [760, 175, 65], "defaults": [60, 25, 15]})


def test_woe_iv_without_smoothing_matches_formula():
    table, iv = woe_iv(TABLE, smoothing=0)
    assert table["WoE"].iloc[0] == pytest.approx(math.log((700 / 900) / (60 / 100)))
    assert iv == pytest.approx(0.1737, abs=1e-3)


def test_woe_iv_default_smoothing_handles_empty_bin():
    table = pd.DataFrame({"bin": ["a", "b"], "N": [10, 5], "defaults": [0, 5]})
    out, iv = woe_iv(table)
    assert np.isfinite(out["WoE"]).all()
    assert iv > 0
