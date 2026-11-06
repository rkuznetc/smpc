import pandas as pd

from conftest import COMMON

from smpc.channel import Channel
from smpc.data import prepare_bank, prepare_telecom
from smpc.parties import Bank, Telecom
from smpc.protocols import PSICAProtocol, PSIProtocol


def test_psi_returns_exact_intersection(bank, telecom):
    result = PSIProtocol().run(bank, telecom)
    assert set(result.bank_output) == COMMON
    # Пересечение частичное: «вернуть всех клиентов банка» было бы ошибкой.
    assert set(result.bank_output) != set(bank.ids)
    assert result.telecom_output is None


def test_psi_ca_returns_only_size(bank, telecom):
    result = PSICAProtocol().run(bank, telecom)
    assert result.bank_output == len(COMMON)


def test_messages_go_through_channel(bank, telecom):
    channel = Channel()
    result = PSIProtocol().run(bank, telecom, channel)
    labels = [m.label for m in channel.log]
    assert labels == ["server_setup", "request", "response"]
    assert result.bytes_sent == channel.bytes_total > 0


def test_empty_intersection():
    bank = Bank(prepare_bank(pd.DataFrame({"SK_ID_CURR": [1, 2, 3], "TARGET": [0, 1, 0]})))
    telecom = Telecom(prepare_telecom(pd.DataFrame({"SK_ID_CURR": [4, 5]})))
    assert PSIProtocol().run(bank, telecom).bank_output == []
    assert PSICAProtocol().run(bank, telecom).bank_output == 0
