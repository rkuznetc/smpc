import pandas as pd
import pytest

from smpc.ml import BankOnlyModel, CeilingModel, ScoringModel, WoEScorecard, compare_models
from smpc.protocols import PS3I, PrivateIntersectionSum


def test_scoring_model_is_abstract():
    with pytest.raises(TypeError):
        ScoringModel()


@pytest.mark.parametrize("model", [BankOnlyModel(), CeilingModel(), WoEScorecard(BankOnlyModel(), {})])
def test_ml_stubs_not_implemented_yet(model):
    with pytest.raises(NotImplementedError):
        model.fit(pd.DataFrame(), pd.Series(dtype=int))


def test_compare_models_not_implemented_yet():
    with pytest.raises(NotImplementedError):
        compare_models([], pd.DataFrame(), pd.Series(dtype=int))


@pytest.mark.parametrize("protocol", [PrivateIntersectionSum("DAYS_LAST_PHONE_CHANGE"), PS3I()])
def test_future_protocols_not_implemented_yet(protocol, bank, telecom):
    with pytest.raises(NotImplementedError):
        protocol.run(bank, telecom)
