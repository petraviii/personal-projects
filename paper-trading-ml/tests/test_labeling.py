"""labeling.py tests: forward returns, binary labels, and walk-forward splits."""

import numpy as np
import pandas as pd
import pytest

from papertrader.labeling import forward_return, make_labels, walk_forward_splits


def test_forward_return_one_day_ahead() -> None:
    close = pd.Series([100.0, 105.0, 110.0, 90.0, 95.0])
    result = forward_return(close, horizon=1)

    expected = [0.05, 110 / 105 - 1, 90 / 110 - 1, 95 / 90 - 1, np.nan]
    assert result.iloc[:-1].tolist() == pytest.approx(expected[:-1])
    assert np.isnan(result.iloc[-1])


def test_forward_return_two_days_ahead_skips_a_day() -> None:
    close = pd.Series([100.0, 105.0, 110.0, 90.0, 95.0])
    result = forward_return(close, horizon=2)

    # close[2]/close[0] - 1, close[3]/close[1] - 1, close[4]/close[2] - 1, then NaN, NaN
    expected = [110 / 100 - 1, 90 / 105 - 1, 95 / 110 - 1]
    assert result.iloc[:3].tolist() == pytest.approx(expected)
    assert result.iloc[3:].isna().all()


def test_make_labels_is_1_when_forward_return_exceeds_threshold() -> None:
    close = pd.Series([100.0, 105.0, 110.0, 90.0, 95.0])
    labels = make_labels(close, horizon=1, threshold=0.0)

    # Forward returns: +5%, +4.76%, -18.2%, +5.6%, NaN -> up, up, down, up, NaN
    assert labels.iloc[:-1].tolist() == [1.0, 1.0, 0.0, 1.0]
    assert np.isnan(labels.iloc[-1])


def test_make_labels_respects_nonzero_threshold() -> None:
    close = pd.Series([100.0, 103.0, 100.0])
    # Forward return for day 0 is +3%, which beats a 5% threshold only if 3% > 5% (false).
    labels = make_labels(close, horizon=1, threshold=0.05)

    assert labels.iloc[0] == 0.0


def test_walk_forward_splits_never_leak_future_into_train() -> None:
    splits = list(walk_forward_splits(n_samples=20, n_splits=4))

    assert len(splits) == 4
    for train_idx, test_idx in splits:
        assert train_idx.max() < test_idx.min()


def test_walk_forward_splits_train_set_expands_each_fold() -> None:
    splits = list(walk_forward_splits(n_samples=20, n_splits=4))

    train_sizes = [len(train_idx) for train_idx, _ in splits]
    assert train_sizes == sorted(train_sizes)  # non-decreasing, expanding window
