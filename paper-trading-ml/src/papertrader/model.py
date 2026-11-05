"""Baseline classifier: LogisticRegression trained/evaluated on walk-forward folds."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

from papertrader.labeling import walk_forward_splits


def prepare_dataset(features: pd.DataFrame, labels: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Align features and labels on their shared index, dropping rows with
    any NaN (indicator warmup at the start, label horizon tail at the end)."""
    combined = features.join(labels.rename("label")).dropna()
    return combined.drop(columns="label"), combined["label"]


def evaluate_walk_forward(
    features: pd.DataFrame, labels: pd.Series, n_splits: int = 5
) -> pd.DataFrame:
    """Train a fresh LogisticRegression on each walk-forward fold's train
    set, score it on that fold's held-out test set. One row per fold.

    A fold is skipped (and noted as such) if its training set contains
    only one class — LogisticRegression can't fit on that, and it can
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

        model = LogisticRegression(max_iter=1000)
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


def train_final_model(features: pd.DataFrame, labels: pd.Series) -> LogisticRegression:
    """Fit a LogisticRegression on the full aligned dataset (for downstream
    use, e.g. generating live signals in the backtest engine)."""
    X, y = prepare_dataset(features, labels)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model
