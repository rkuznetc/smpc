import pytest

from smpc import reference
from smpc.channel import Channel
from smpc.protocols import PISPrototype


@pytest.mark.parametrize("feature", ["DAYS_LAST_PHONE_CHANGE", "DEF_30_CNT_SOCIAL_CIRCLE"])
def test_pis_prototype_matches_reference(bank, telecom, feature):
    result = PISPrototype(feature=feature, key_bits=1024).run(bank, telecom)
    expected = reference.bin_stats(bank, telecom, feature)
    got = result.bank_output[["bin", "N", "defaults"]].reset_index(drop=True)
    assert got.equals(expected.reset_index(drop=True))
    assert result.details["homomorphic_check"] is True


def test_pis_prototype_message_flow(bank, telecom):
    channel = Channel()
    PISPrototype(feature="DAYS_LAST_PHONE_CHANGE", key_bits=1024).run(bank, telecom, channel)
    assert [(m.sender, m.receiver, m.label) for m in channel.log] == [
        ("bank", "telecom", "hmac_tokens"),
        ("bank", "demo_env", "paillier_public_key"),
        ("demo_env", "bank", "encrypted_aggregates"),
    ]
    # 10 псевдонимов банка по 32 байта.
    assert channel.log[0].size == 10 * 32


def test_pis_prototype_declares_its_leakage():
    leakage = PISPrototype(feature="DAYS_LAST_PHONE_CHANGE").leakage
    assert "demo_env" in leakage
    assert "совпали" in leakage["telecom"]
