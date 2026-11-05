"""Forward-return binary labeling and a chronological walk-forward split helper."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    """Percentage return from today's close to the close `horizon` periods ahead."""
    return close.shift(-horizon) / close - 1


def make_labels(close: pd.Series, horizon: int = 5, threshold: float = 0.0) -> pd.Series:
    """Binary up/down label: 1 if the forward `horizon`-day return exceeds
    `threshold`, else 0. The last `horizon` rows have no forward price yet
    and are labeled NaN rather than guessed."""
    fwd = forward_return(close, horizon)
    labels = (fwd > threshold).astype(float)
    labels[fwd.isna()] = np.nan
    return labels


def walk_forward_splits(
    n_samples: int, n_splits: int = 5
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Chronological (expanding-window) train/test index splits.

    Train is always strictly before test and folds are never shuffled, so
    no future data can leak into training — the standard correctness
    requirement for time-series ML. Thin wrapper around sklearn's
    TimeSeriesSplit, kept here so callers only need papertrader imports.
    """
    splitter = TimeSeriesSplit(n_splits=n_splits)
    yield from splitter.split(np.zeros(n_samples))
