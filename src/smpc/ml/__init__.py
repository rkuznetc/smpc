"""Скелет ML (без реализации): интерфейс модели и три планируемые модели."""

from smpc.ml.base import BankOnlyModel, CeilingModel, ScoringModel, WoEScorecard, compare_models

__all__ = ["ScoringModel", "BankOnlyModel", "CeilingModel", "WoEScorecard", "compare_models"]
