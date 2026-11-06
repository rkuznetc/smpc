"""Протоколы. Новый метод = новый подкласс Protocol + запись в REGISTRY."""

from smpc.protocols.base import Protocol, ProtocolResult
from smpc.protocols.future import PS3I, PrivateIntersectionSum
from smpc.protocols.openmined import PSICAProtocol, PSIProtocol
from smpc.protocols.pis_prototype import PISPrototype

#: Реализованные протоколы, доступные в демо (ключ -> класс).
REGISTRY = {
    "psi": PSIProtocol,
    "psi_ca": PSICAProtocol,
    "pis": PISPrototype,
}

__all__ = [
    "Protocol",
    "ProtocolResult",
    "PSIProtocol",
    "PSICAProtocol",
    "PISPrototype",
    "PrivateIntersectionSum",
    "PS3I",
    "REGISTRY",
]
