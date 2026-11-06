import numpy as np
import pandas as pd
import pytest

from smpc.data import assign_bins, load_bank, load_telecom, prepare_bank, subsample


def test_bins_days_last_phone_change():
    s = pd.Series([-100, -400, -2000, np.nan, 5, 0])
    assert assign_bins(s, "DAYS_LAST_PHONE_CHANGE").tolist() == [
        "0–1 год", "1–3 года", "более 3 лет", "Пропуск", "Пропуск", "0–1 год",
    ]


def test_bins_social_circle_counts():
    s = pd.Series([0, 1, 2, 7, np.nan, -1])
    assert assign_bins(s, "DEF_30_CNT_SOCIAL_CIRCLE").tolist() == ["0", "1", "2+", "2+", "Пропуск", "Пропуск"]


def test_prepare_bank_rejects_conflicting_target():
    frame = pd.DataFrame({"SK_ID_CURR": [1, 1], "TARGET": [0, 1]})
    with pytest.raises(ValueError):
        prepare_bank(frame)


def test_prepare_bank_rejects_non_binary_target():
    frame = pd.DataFrame({"SK_ID_CURR": [1, 2], "TARGET": [0, 2]})
    with pytest.raises(ValueError):
        prepare_bank(frame)


def test_prepare_bank_drops_duplicates_and_stringifies_ids():
    frame = pd.DataFrame({"SK_ID_CURR": [5, 5, 7], "TARGET": [1, 1, 0]})
    out = prepare_bank(frame)
    assert out["ID"].tolist() == ["5", "7"]
    assert out["TARGET"].tolist() == [1, 0]


def test_load_roundtrip_from_parquet(tmp_path):
    pd.DataFrame({"SK_ID_CURR": [10, 11], "TARGET": [0, 1], "EXTRA": [1.0, 2.0]}).to_parquet(tmp_path / "b.parquet")
    pd.DataFrame({"SK_ID_CURR": [11, 12], "DEF_30_CNT_SOCIAL_CIRCLE": [0.0, 1.0]}).to_parquet(tmp_path / "t.parquet")
    bank = load_bank(tmp_path / "b.parquet")
    telecom = load_telecom(tmp_path / "t.parquet", ["DEF_30_CNT_SOCIAL_CIRCLE"])
    assert list(bank.columns) == ["ID", "TARGET"]
    assert list(telecom.columns) == ["ID", "DEF_30_CNT_SOCIAL_CIRCLE"]
    assert telecom["ID"].tolist() == ["11", "12"]


def test_load_telecom_missing_feature(tmp_path):
    pd.DataFrame({"SK_ID_CURR": [1]}).to_parquet(tmp_path / "t.parquet")
    with pytest.raises(Exception):
        load_telecom(tmp_path / "t.parquet", ["DEF_30_CNT_SOCIAL_CIRCLE"])


def test_subsample():
    frame = pd.DataFrame({"x": range(100)})
    assert len(subsample(frame, 10, seed=1)) == 10
    assert subsample(frame, None) is frame
