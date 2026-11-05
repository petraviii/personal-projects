"""features.py tests against small series with hand-computed expected values."""

import numpy as np
import pandas as pd
import pytest

from papertrader.features import (
    build_feature_matrix,
    daily_returns,
    ema,
    lagged_returns,
    macd,
    rolling_volatility,
    rsi,
    sma,
)


def test_sma_matches_hand_computed_window_average() -> None:
    close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = sma(close, window=3)

    # First two entries have fewer than `window` observations -> NaN.
    assert result.iloc[:2].isna().all()
    # (1+2+3)/3, (2+3+4)/3, (3+4+5)/3
    assert result.iloc[2:].tolist() == [2.0, 3.0, 4.0]


def test_ema_matches_hand_computed_recursive_average() -> None:
    close = pd.Series([10.0, 20.0, 30.0])
    result = ema(close, span=2)  # alpha = 2 / (span + 1) = 2/3

    # y0 = x0; y_t = (1 - alpha) * y_{t-1} + alpha * x_t
    expected = [10.0, (1 / 3) * 10 + (2 / 3) * 20, (1 / 3) * 16.666666 + (2 / 3) * 30]
    assert result.tolist() == pytest.approx(expected, rel=1e-4)


def test_rsi_is_100_when_window_has_only_gains() -> None:
    close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])  # strictly increasing
    result = rsi(close, window=3)

    assert result.iloc[3:].tolist() == pytest.approx([100.0, 100.0])


def test_rsi_is_50_on_a_perfectly_flat_window() -> None:
    close = pd.Series([5.0, 5.0, 5.0, 5.0])  # no gains or losses at all
    result = rsi(close, window=3)

    assert result.iloc[-1] == pytest.approx(50.0)


def test_macd_is_difference_of_our_own_emas() -> None:
    close = pd.Series([float(x) for x in range(1, 40)])
    result = macd(close, fast=5, slow=10, signal=3)

    expected_macd_line = ema(close, 5) - ema(close, 10)
    assert result["macd"].tolist() == pytest.approx(expected_macd_line.tolist())
    assert result["histogram"].tolist() == pytest.approx(
        (result["macd"] - result["signal"]).tolist()
    )


def test_daily_returns_matches_pct_change_by_hand() -> None:
    close = pd.Series([100.0, 110.0, 99.0])
    result = daily_returns(close)

    assert result.iloc[0] is not None and np.isnan(result.iloc[0])
    assert result.iloc[1] == pytest.approx(0.10)
    assert result.iloc[2] == pytest.approx(-0.10)


def test_rolling_volatility_is_zero_for_constant_returns() -> None:
    # Each step is +10% of the previous close -> constant daily return.
    close = pd.Series([100.0 * (1.1**i) for i in range(5)])
    result = rolling_volatility(close, window=3)

    assert result.iloc[-1] == pytest.approx(0.0, abs=1e-9)


def test_lagged_returns_matches_pct_change_periods() -> None:
    close = pd.Series([100.0, 110.0, 121.0, 108.9])
    result = lagged_returns(close, lags=[1, 2])

    assert result["return_1d"].iloc[1] == pytest.approx(0.10)
    # (121 - 100) / 100 == 0.21 over two periods
    assert result["return_2d"].iloc[2] == pytest.approx(0.21)


def test_build_feature_matrix_has_expected_columns_and_length() -> None:
    close = pd.Series([100.0 + i for i in range(60)])
    result = build_feature_matrix(close)

    expected_columns = {
        "sma_10",
        "sma_50",
        "ema_12",
        "ema_26",
        "rsi_14",
        "macd",
        "signal",
        "histogram",
        "volatility_20",
        "return_1d",
        "return_5d",
        "return_10d",
    }
    assert set(result.columns) == expected_columns
    assert len(result) == len(close)
