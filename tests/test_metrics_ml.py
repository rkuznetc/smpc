import numpy as np

from smpc.metrics import best_f1_threshold, classification_row


def test_best_f1_threshold_maximises_f1_on_validation():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.6, 0.7, 0.8, 0.9])
    threshold = best_f1_threshold(y, p)
    predicted = (p >= threshold).astype(int)
    assert (predicted == y).all()


def test_classification_row_has_expected_keys_and_values():
    y = np.array([0, 1, 1, 0])
    p = np.array([0.2, 0.9, 0.4, 0.1])
    row = classification_row(y, p, threshold=0.5)
    assert {"ROC-AUC", "PR-AUC", "Log-loss", "Brier", "Precision", "Recall", "F1", "Balanced accuracy"} <= set(row)
    assert row["ROC-AUC"] == 1.0
    assert row["Recall"] == 0.5
