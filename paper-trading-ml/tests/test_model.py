"""model.py tests against a small, perfectly-separable synthetic dataset."""

import pandas as pd
import pytest

from papertrader.model import evaluate_walk_forward, prepare_dataset, train_final_model


def _separable_dataset(n: int = 40) -> tuple[pd.DataFrame, pd.Series]:
    # Alternating +1/-1 feature with a label that exactly matches its sign,
    # so every contiguous slice has both classes and is trivially learnable.
    x = [1.0 if i % 2 == 0 else -1.0 for i in range(n)]
    features = pd.DataFrame({"x": x})
    labels = pd.Series([1.0 if v > 0 else 0.0 for v in x])
    return features, labels


def test_prepare_dataset_drops_rows_with_any_nan() -> None:
    features = pd.DataFrame({"x": [1.0, 2.0, float("nan"), 4.0]})
    labels = pd.Series([1.0, 0.0, 1.0, float("nan")])

    X, y = prepare_dataset(features, labels)

    # Rows 2 (NaN feature) and 3 (NaN label) are dropped; 0 and 1 remain.
    assert list(X.index) == [0, 1]
    assert list(y.index) == [0, 1]


def test_evaluate_walk_forward_achieves_high_accuracy_on_separable_data() -> None:
    features, labels = _separable_dataset(n=40)
    results = evaluate_walk_forward(features, labels, n_splits=4)

    assert len(results) == 4
    evaluated = results[~results["skipped"]]
    assert len(evaluated) > 0
    assert (evaluated["accuracy"] == 1.0).all()


def test_evaluate_walk_forward_marks_single_class_fold_as_skipped() -> None:
    # Every training window has only label 0 -> can't fit a 2-class model.
    features = pd.DataFrame({"x": [1.0] * 10 + [-1.0, 1.0] * 5})
    labels = pd.Series([0.0] * 10 + [0.0, 1.0] * 5)
    results = evaluate_walk_forward(features, labels, n_splits=5)

    assert results["skipped"].any()
    skipped_rows = results[results["skipped"]]
    assert skipped_rows["accuracy"].isna().all()


def test_train_final_model_can_predict() -> None:
    features, labels = _separable_dataset(n=20)
    model = train_final_model(features, labels)

    predictions = model.predict(features)
    assert list(predictions) == pytest.approx(labels.tolist())
