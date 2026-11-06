"""PSI и PSI-CA на библиотеке OpenMined PSI (уровень 1).

Логика как в ноутбуках 01 и 02: банк — клиент, телеком — сервер,
fpr = 0.001, структура RAW. Флаг reveal_intersection:
    True  -> PSI: банк получает индексы совпавших ID;
    False -> PSI-CA: банк получает только число совпадений.

Сообщения OpenMined (protobuf) передаются через Channel в байтах:
ServerSetup и Response от телекома банку, Request от банка телекому.
"""

from __future__ import annotations

from typing import Any

import private_set_intersection.python as psi

from smpc.channel import Channel
from smpc.parties import Bank, Telecom
from smpc.protocols.base import Protocol

_DATA_STRUCTURES = {
    "RAW": psi.DataStructure.RAW,
    "GCS": psi.DataStructure.GCS,
    "BLOOM_FILTER": psi.DataStructure.BLOOM_FILTER,
}


class _OpenMinedPSI(Protocol):
    reveal_intersection: bool = True

    def __init__(self, fpr: float = 0.001, data_structure: str = "RAW") -> None:
        if data_structure not in _DATA_STRUCTURES:
            raise ValueError(f"data_structure: одно из {sorted(_DATA_STRUCTURES)}")
        self.fpr = fpr
        self.data_structure = data_structure

    def _exchange(self, bank: Bank, telecom: Telecom, channel: Channel):
        bank_ids = bank.ids
        telecom_ids = telecom.ids

        # Секретные ключи генерируются внутри клиента и сервера.
        bank_client = psi.client.CreateWithNewKey(self.reveal_intersection)
        telecom_server = psi.server.CreateWithNewKey(self.reveal_intersection)

        # Телеком: защищённое представление своей базы -> банку.
        setup = telecom_server.CreateSetupMessage(
            self.fpr, len(bank_ids), telecom_ids, _DATA_STRUCTURES[self.data_structure]
        )
        setup_bytes = channel.send("telecom", "bank", "server_setup", setup.SerializeToString())

        # Банк: зашифрованный запрос -> телекому.
        request = bank_client.CreateRequest(bank_ids)
        request_bytes = channel.send("bank", "telecom", "request", request.SerializeToString())

        # Телеком: применяет свой ключ к запросу банка -> банку.
        response = telecom_server.ProcessRequest(psi.Request.FromString(request_bytes))
        response_bytes = channel.send("telecom", "bank", "response", response.SerializeToString())

        return (
            bank_client,
            bank_ids,
            psi.ServerSetup.FromString(setup_bytes),
            psi.Response.FromString(response_bytes),
        )


class PSIProtocol(_OpenMinedPSI):
    """PSI: банк получает список общих ID."""

    name = "PSI"
    reveal_intersection = True
    leakage = {
        "bank": "список общих ID (пересечение)",
        "telecom": "только размер множества банка",
    }

    def _execute(self, bank: Bank, telecom: Telecom, channel: Channel) -> tuple[Any, Any, dict[str, Any]]:
        client, bank_ids, setup, response = self._exchange(bank, telecom, channel)
        indices = client.GetIntersection(setup, response)
        common = sorted(bank_ids[i] for i in indices)
        return common, None, {"n_common": len(common), "data_structure": self.data_structure}


class PSICAProtocol(_OpenMinedPSI):
    """PSI-CA: банк получает только число общих ID."""

    name = "PSI-CA"
    reveal_intersection = False
    leakage = {
        "bank": "только число общих ID",
        "telecom": "только размер множества банка",
    }

    def _execute(self, bank: Bank, telecom: Telecom, channel: Channel) -> tuple[Any, Any, dict[str, Any]]:
        client, _, setup, response = self._exchange(bank, telecom, channel)
        size = int(client.GetIntersectionSize(setup, response))
        return size, None, {"data_structure": self.data_structure}
