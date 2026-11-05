"""Hand-rolled technical indicators computed from a close-price series.

Kept dependency-free (pure pandas/numpy) rather than pulling in an
indicator library — these are simple enough to implement directly, and
doing so keeps the math auditable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    """Simple moving average over `window` periods."""
    return close.rolling(window=window).mean()


def ema(close: pd.Series, span: int) -> pd.Series:
    """Exponential moving average with the given span (pandas ewm, adjust=False)."""
    return close.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index via a simple rolling mean of gains/losses.

    100 when there have been no losses in the window (and some gains);
    50 when the window is perfectly flat (no gains or losses) — both are
    edge cases where the textbook gain/loss ratio is undefined.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    result = 100 - (100 / (1 + rs))
    result = result.where(avg_loss != 0, other=np.where(avg_gain > 0, 100.0, 50.0))
    return result


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """MACD line, signal line, and histogram (macd - signal)."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": histogram}
    )


def daily_returns(close: pd.Series) -> pd.Series:
    """Simple day-over-day percentage return."""
    return close.pct_change()


def lagged_returns(close: pd.Series, lags: list[int]) -> pd.DataFrame:
    """Percentage return over each of the given lag periods, as separate columns."""
    return pd.DataFrame({f"return_{lag}d": close.pct_change(periods=lag) for lag in lags})


def rolling_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """Rolling standard deviation of daily returns over `window` periods."""
    return daily_returns(close).rolling(window=window).std()


def build_feature_matrix(close: pd.Series) -> pd.DataFrame:
    """Combine all indicators into a single feature DataFrame indexed by date."""
    features = pd.DataFrame(index=close.index)
    features["sma_10"] = sma(close, 10)
    features["sma_50"] = sma(close, 50)
    features["ema_12"] = ema(close, 12)
    features["ema_26"] = ema(close, 26)
    features["rsi_14"] = rsi(close, 14)
    features = features.join(macd(close))
    features["volatility_20"] = rolling_volatility(close, 20)
    features = features.join(lagged_returns(close, [1, 5, 10]))
    return features
