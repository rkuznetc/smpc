import pandas as pd
import pytest

from smpc.ml import ScoringModel, WoEScorecard, BankOnlyModel
from smpc.protocols import PS3I, PrivateIntersectionSum


def test_scoring_model_is_abstract():
    with pytest.raises(TypeError):
        ScoringModel()


def test_woe_scorecard_stub_not_implemented_yet():
    with pytest.raises(NotImplementedError):
        WoEScorecard(BankOnlyModel(), {}).fit(pd.DataFrame(), pd.Series(dtype=int))


@pytest.mark.parametrize("protocol", [PrivateIntersectionSum("DAYS_LAST_PHONE_CHANGE"), PS3I()])
def test_future_protocols_not_implemented_yet(protocol, bank, telecom):
    with pytest.raises(NotImplementedError):
        protocol.run(bank, telecom)
