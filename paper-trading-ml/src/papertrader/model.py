"""Classifiers trained/evaluated on walk-forward folds: a LogisticRegression
baseline, pluggable alternatives (e.g. RandomForest), and walk-forward
grid-search tuning."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Any

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

from papertrader.labeling import walk_forward_splits

ModelFactory = Callable[[], ClassifierMixin]


def logistic_regression_factory(**params: Any) -> LogisticRegression:
    return LogisticRegression(max_iter=1000, **params)


def random_forest_factory(**params: Any) -> RandomForestClassifier:
    return RandomForestClassifier(random_state=42, **params)


def prepare_dataset(features: pd.DataFrame, labels: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Align features and labels on their shared index, dropping rows with
    any NaN (indicator warmup at the start, label horizon tail at the end)."""
    combined = features.join(labels.rename("label")).dropna()
    return combined.drop(columns="label"), combined["label"]


def evaluate_walk_forward(
    features: pd.DataFrame,
    labels: pd.Series,
    n_splits: int = 5,
    model_factory: ModelFactory = logistic_regression_factory,
) -> pd.DataFrame:
    """Train a fresh model (from `model_factory`, LogisticRegression by
    default) on each walk-forward fold's train set, score it on that fold's
    held-out test set. One row per fold.

    A fold is skipped (and noted as such) if its training set contains
    only one class — most classifiers can't fit on that, and it can
    happen early on with a small expanding window.
    """
    X, y = prepare_dataset(features, labels)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(walk_forward_splits(len(X), n_splits=n_splits)):
        y_train = y.iloc[train_idx]
        if y_train.nunique() < 2:
            rows.append(
                {"fold": fold, "train_size": len(train_idx), "test_size": len(test_idx),
                 "skipped": True, "accuracy": None, "precision": None, "recall": None}
            )
            continue

        model = model_factory()
        model.fit(X.iloc[train_idx], y_train)
        y_pred = model.predict(X.iloc[test_idx])
        y_test = y.iloc[test_idx]

        rows.append(
            {
                "fold": fold,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "skipped": False,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def walk_forward_predictions(
    features: pd.DataFrame,
    labels: pd.Series,
    n_splits: int = 5,
    model_factory: ModelFactory = logistic_regression_factory,
) -> pd.Series:
    """Out-of-sample predictions for each walk-forward test fold, stitched
    into one series aligned to the prepared dataset's index — a lookahead-free
    signal suitable for backtesting.

    Rows in a skipped fold (training set with only one class) are left as
    NaN rather than guessed.
    """
    X, y = prepare_dataset(features, labels)
    predictions = pd.Series(index=X.index, dtype=float)
    for train_idx, test_idx in walk_forward_splits(len(X), n_splits=n_splits):
        y_train = y.iloc[train_idx]
        if y_train.nunique() < 2:
            continue
        model = model_factory()
        model.fit(X.iloc[train_idx], y_train)
        predictions.iloc[test_idx] = model.predict(X.iloc[test_idx])
    return predictions


def train_final_model(
    features: pd.DataFrame,
    labels: pd.Series,
    model_factory: ModelFactory = logistic_regression_factory,
) -> ClassifierMixin:
    """Fit a model on the full aligned dataset (for downstream use, e.g.
    generating live signals in the backtest engine)."""
    X, y = prepare_dataset(features, labels)
    model = model_factory()
    model.fit(X, y)
    return model


def grid_search_walk_forward(
    features: pd.DataFrame,
    labels: pd.Series,
    model_factory: Callable[..., ClassifierMixin],
    param_grid: Mapping[str, Sequence[Any]],
    n_splits: int = 5,
) -> pd.DataFrame:
    """Score every combination in `param_grid` (e.g.
    `{"n_estimators": [50, 100], "max_depth": [3, None]}`) by its mean
    accuracy across walk-forward folds (skipped folds excluded).

    Each combination gets a fresh model per fold, built as
    `model_factory(**params)` — the same walk-forward evaluation
    `evaluate_walk_forward` does, just repeated per combination. Returns
    one row per combination, sorted by mean accuracy descending.
    """
    keys = list(param_grid.keys())
    rows = []
    for values in itertools.product(*param_grid.values()):
        params = dict(zip(keys, values))
        results = evaluate_walk_forward(
            features, labels, n_splits=n_splits, model_factory=partial(model_factory, **params)
        )
        evaluated = results[~results["skipped"]]
        rows.append(
            {
                **params,
                "mean_accuracy": evaluated["accuracy"].mean() if len(evaluated) else float("nan"),
                "mean_precision": evaluated["precision"].mean() if len(evaluated) else float("nan"),
                "mean_recall": evaluated["recall"].mean() if len(evaluated) else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_accuracy", ascending=False, ignore_index=True)
