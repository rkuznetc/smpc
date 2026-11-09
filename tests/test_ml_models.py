import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from smpc.ml import ApproxLogisticModel, BankOnlyModel, CeilingModel, WoEScorecard, compare_models
from smpc.ml.approx import plain_gradient


def _toy(n=150, seed=0):
    rng = np.random.default_rng(seed)
    X = sp.csr_matrix(rng.normal(size=(n, 3)))
    y = (X[:, 0].toarray().ravel() > 0).astype(int)
    return X, y


def test_bank_only_learns_signal_and_outputs_probabilities():
    X, y = _toy()
    model = BankOnlyModel(epochs=50, learning_rate=0.5).fit(X, y)
    proba = model.predict_proba(X)
    assert model.name == "bank_only"
    assert np.all((proba > 0) & (proba < 1))
    assert ((proba > 0.5) == y).mean() > 0.8


def test_ceiling_uses_the_same_algorithm_under_its_own_name():
    assert CeilingModel().name == "ceiling"
    assert issubclass(CeilingModel, ApproxLogisticModel)


def test_predict_before_fit_is_an_error():
    with pytest.raises(RuntimeError, match="не обучена"):
        BankOnlyModel().predict_proba(sp.csr_matrix(np.ones((2, 2))))


def test_custom_gradient_is_called_each_epoch():
    calls = []

    def spy(w, b, X, y):
        calls.append(1)
        return plain_gradient(w, b, X, y)

    X, y = _toy(n=30)
    ApproxLogisticModel(gradient=spy, epochs=3).fit(X, y)
    assert len(calls) == 3


def test_compare_models_returns_one_row_per_model():
    X, y = _toy()
    bank = BankOnlyModel(epochs=20).fit(X[:, :1], y)
    ceiling = CeilingModel(epochs=20).fit(X, y)
    table = compare_models(
        {"Только банк": (bank, X[:, :1]), "Потолок": (ceiling, X)},
        y,
    )
    assert list(table.index) == ["Только банк", "Потолок"]
    assert list(table["Признаков"]) == [1, 3]
    assert table["ROC-AUC"].between(0, 1).all()


def test_woe_scorecard_is_still_a_stub():
    with pytest.raises(NotImplementedError):
        WoEScorecard(BankOnlyModel(), {}).fit(pd.DataFrame(), pd.Series(dtype=int))
