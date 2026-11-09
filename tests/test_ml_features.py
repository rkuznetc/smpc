import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from smpc.ml.features import (
    build_block,
    clean_frame,
    clip_sparse,
    is_id_like,
    l1_logistic,
    l1_select,
    SplitIndices,
    make_splits,
    select_feature_columns,
)

def test_id_like_columns_are_excluded():
    columns = ["ID", "SK_ID_CURR", "SK_ID_BUREAU", "TARGET", "AMT", "X_ID", "INDEX", "NAME"]
    assert select_feature_columns(pd.DataFrame(columns=columns)) == ["AMT", "NAME"]
    assert is_id_like("row_id") is True  # служебное имя из списка
    assert is_id_like("CLIENT_ID") and is_id_like("SK_ID_PREV")


def test_clean_frame_numbers_and_categories():
    frame = pd.DataFrame({"num": [1.0, np.inf, np.nan], "cat": ["a", None, "b"]})
    out = clean_frame(frame)
    assert out["num"].isna().tolist() == [False, True, True]
    assert out["cat"].tolist() == ["a", "__MISSING__", "b"]


def test_splits_are_disjoint_and_stratified():
    y = np.array([1] * 100 + [0] * 900)
    splits = make_splits(y, seed=7)
    all_idx = np.concatenate([splits.train, splits.val, splits.test])
    assert len(set(all_idx.tolist())) == len(y)
    for part in (splits.train, splits.val, splits.test):
        assert y[part].mean() == pytest.approx(0.1, abs=0.02)


def test_block_is_fitted_on_train_only():
    frame = pd.DataFrame({"AMT": [1.0, 2.0, 3.0, 4.0, 5.0, 3.0]})
    frame["TARGET"] = [0, 0, 0, 0, 0, 0]
    splits = SplitIndices(train=np.arange(5), val=np.array([4]), test=np.array([5]))
    block = build_block(frame, "bank__", splits)
    # Среднее train = 3: тестовое значение 3 должно получить 0, а не значение,
    # если бы scaler учился и на тестовой строке.
    assert block.test.toarray()[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert block.names == ["bank__numeric__AMT"]


def test_clip_limits_nonzero_values():
    m = sp.csr_matrix(np.array([[0.0, 9.0, -7.0], [0.5, 0.0, 0.0]]))
    out = clip_sparse(m.copy())
    assert out.toarray().max() == 5.0 and out.toarray().min() == -5.0
    assert out.nnz == 3  # нули не появляются и не исчезают


def test_l1_select_limits_count_and_sorts_indices():
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.normal(size=(300, 6)))
    y = (X[:, 0].toarray().ravel() + 0.5 * X[:, 3].toarray().ravel() > 0).astype(int)
    names = [f"f{j}" for j in range(6)]
    selected, importance = l1_select(X, y, names, max_features=2, C=0.5)
    assert len(selected) <= 2
    assert list(selected) == sorted(selected)
    assert importance.shape == (6,)


def test_l1_select_raises_when_everything_is_zero():
    X = sp.csr_matrix(np.random.default_rng(0).normal(size=(50, 3)))
    y = (X[:, 0].toarray().ravel() > 0).astype(int)
    with pytest.raises(ValueError, match="обнулил все признаки"):
        l1_select(X, y, ["a", "b", "c"], max_features=None, C=1e-9)


def test_l1_select_checks_name_count():
    X = sp.csr_matrix(np.ones((4, 2)))
    with pytest.raises(ValueError):
        l1_select(X, np.array([0, 1, 0, 1]), ["only_one"], max_features=None)


def test_l1_logistic_emits_no_penalty_deprecation_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        l1_logistic(C=0.1, seed=0).fit(sp.csr_matrix(np.eye(4)), np.array([0, 1, 0, 1]))
    messages = [str(w.message) for w in caught if "penalty" in str(w.message)]
    assert messages == []
