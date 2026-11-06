"""Paper-trading simulation: turn a daily long/flat signal into a simulated
portfolio equity curve, plus the standard summary metrics."""

from __future__ import annotations

import pandas as pd


def run_backtest(
    close: pd.Series,
    signal: pd.Series,
    position_size: float = 1.0,
    commission_bps: float = 0.0,
    initial_capital: float = 10_000.0,
) -> pd.DataFrame:
    """Simulate holding `position_size` of capital in the asset whenever
    `signal` is 1 (flat otherwise), earning the return from day t to t+1.

    `signal[t]` is a decision made with data available at day t, so it
    earns the close[t] -> close[t+1] return (aligned via shift(-1)) — the
    same convention `labeling.make_labels` uses, avoiding lookahead. The
    last row has no next-day price yet, so it earns nothing.

    A flat `commission_bps` cost is charged on any change in position
    size (entering, exiting, or resizing), as a simple trading-cost
    assumption.
    """
    asset_return = close.pct_change().shift(-1)
    position = signal.astype(float) * position_size

    gross_return = position * asset_return

    position_change = position.diff().abs()
    position_change.iloc[0] = abs(position.iloc[0])  # entering the first position is also a trade
    cost = position_change * (commission_bps / 10_000.0)

    daily_return = (gross_return - cost).fillna(0.0)
    equity = initial_capital * (1 + daily_return).cumprod()

    return pd.DataFrame({"position": position, "daily_return": daily_return, "equity": equity})


def compute_metrics(
    daily_return: pd.Series, position: pd.Series, trading_days_per_year: int = 252
) -> dict[str, float]:
    """Summarize a daily-return series: total return, CAGR, annualized
    Sharpe ratio, max drawdown, and win rate on invested days.

    "Win rate" here means the fraction of days with `position != 0` that
    had a positive return — a simple per-day measure, not a round-trip
    trade win rate.
    """
    n_days = len(daily_return)
    growth = (1 + daily_return).cumprod()

    total_return = growth.iloc[-1] - 1 if n_days > 0 else 0.0
    cagr = (1 + total_return) ** (trading_days_per_year / n_days) - 1 if n_days > 0 else 0.0

    std = daily_return.std()
    sharpe = (daily_return.mean() / std) * (trading_days_per_year**0.5) if std else 0.0

    running_max = growth.cummax()
    drawdown = (growth - running_max) / running_max
    max_drawdown = drawdown.min() if n_days > 0 else 0.0

    invested_returns = daily_return[position != 0]
    win_rate = (invested_returns > 0).mean() if len(invested_returns) > 0 else 0.0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
    }
