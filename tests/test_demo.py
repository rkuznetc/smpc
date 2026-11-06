from conftest import bank_frame, telecom_frame

from smpc.demo import main


def test_demo_end_to_end(tmp_path, capsys):
    bank_frame().to_parquet(tmp_path / "bank_psi.parquet")
    telecom_frame().to_parquet(tmp_path / "telecom_train.parquet")
    code = main(["--data-dir", str(tmp_path), "--key-bits", "1024"])
    out = capsys.readouterr().out
    assert code == 0
    assert "PSI-CA" in out and "PIS (прототип)" in out
    assert "НЕТ" not in out
