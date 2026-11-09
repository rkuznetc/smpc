import pandas as pd
import pytest

from fixtures_ml import write_parquets
from smpc.experiments.bank_vs_joint import BANK_ONLY, BANK_PLUS_TELECOM, ExperimentConfig, main, run
from smpc.ml.checkpoint import CheckpointMismatch

COMMON = 180  # n=200 банка, без последних 20 в телекоме


@pytest.fixture
def data_dir(tmp_path):
    return write_parquets(tmp_path / "data")


def _config(data_dir, **overrides):
    base = dict(data_dir=data_dir, max_features=6, epochs=3, backends=("open",), seed=42)
    base.update(overrides)
    return ExperimentConfig(**base)


def test_open_run_compares_bank_and_joint_models(data_dir):
    result = run(_config(data_dir))
    assert result.n_common == COMMON
    assert list(result.table["Модель"]) == [BANK_ONLY, BANK_PLUS_TELECOM]
    assert result.table["ROC-AUC"].between(0, 1).all()
    joint = result.table[result.table["Модель"] == BANK_PLUS_TELECOM]["Признаков"].item()
    bank = result.table[result.table["Модель"] == BANK_ONLY]["Признаков"].item()
    assert joint == bank + 5  # все признаки телекома добавляются целиком
    assert set(result.deltas) == {"open"}
    assert all(f.startswith(("bank__", "telecom__")) for f in result.selected_features)


def test_ckks_matches_open_run_and_resumes_from_checkpoint(data_dir, tmp_path):
    pytest.importorskip("tenseal")
    ckpt = tmp_path / "ckpt"
    first = run(_config(data_dir, backends=("open", "ckks"), epochs=1, check_rows=32, checkpoint_dir=ckpt))

    open_rows = first.table[first.table["Backend"] == "open"].set_index("Модель")
    ckks_rows = first.table[first.table["Backend"] == "ckks"].set_index("Модель")
    for label in (BANK_ONLY, BANK_PLUS_TELECOM):
        assert ckks_rows.loc[label, "ROC-AUC"] == pytest.approx(open_rows.loc[label, "ROC-AUC"], abs=1e-3)
    for key in ("bank_only", "bank_plus_telecom"):
        assert first.details["backends"]["ckks"][key]["weight_error"] < 1e-3
        assert (ckpt / f"ckks_{key}.npz").exists()

    # Продолжение обучения с той же контрольной точки: до 3 эпох, а не с нуля.
    second = run(_config(data_dir, backends=("ckks",), epochs=3, check_rows=32, checkpoint_dir=ckpt))
    assert set(second.table["Эпох"]) == {3}


def test_checkpoint_from_other_features_is_rejected(data_dir, tmp_path):
    pytest.importorskip("tenseal")
    ckpt = tmp_path / "ckpt"
    run(_config(data_dir, backends=("ckks",), epochs=1, check_rows=32, checkpoint_dir=ckpt))
    with pytest.raises(CheckpointMismatch):
        # batch_size входит в отпечаток: другое значение — другой эксперимент.
        run(_config(data_dir, backends=("ckks",), epochs=1, batch_size=64, check_rows=32, checkpoint_dir=ckpt))


def test_empty_intersection_is_reported(tmp_path):
    bank = pd.DataFrame({"SK_ID_CURR": [1, 2], "TARGET": [0, 1], "X": [1.0, 2.0]})
    telecom = pd.DataFrame({"SK_ID_CURR": [9], "DAYS_LAST_PHONE_CHANGE": [-5.0]})
    bank.to_parquet(tmp_path / "bank_psi.parquet")
    telecom.to_parquet(tmp_path / "telecom_train.parquet")
    with pytest.raises(ValueError, match="Пересечение пустое"):
        run(_config(tmp_path))


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="Неизвестные backend"):
        ExperimentConfig(backends=("homomorphic-magic",))


def test_cli_prints_table_and_writes_csv(data_dir, tmp_path, capsys):
    out = tmp_path / "results" / "compare.csv"
    code = main(["--data-dir", str(data_dir), "--max-features", "6", "--epochs", "2", "--out", str(out)])
    text = capsys.readouterr().out
    assert code == 0
    assert "Прирост от телекома" in text
    assert out.exists()
    assert len(pd.read_csv(out)) == 2
