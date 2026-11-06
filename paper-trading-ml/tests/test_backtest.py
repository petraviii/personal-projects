"""backtest.py tests against small scripted series with hand-computed expectations."""

import math

import pandas as pd
import pytest

from papertrader.backtest import compute_metrics, run_backtest


def test_run_backtest_matches_hand_computed_returns_with_no_commission() -> None:
    close = pd.Series([100.0, 110.0, 99.0, 108.9])
    signal = pd.Series([1, 0, 1, 0])

    result = run_backtest(close, signal, commission_bps=0.0, initial_capital=10_000.0)

    # Day 0->1: +10% while invested; day 1->2: flat (signal 0); day 2->3: +10% while invested;
    # last day has no further price, so it earns nothing.
    assert result["daily_return"].tolist() == pytest.approx([0.10, 0.0, 0.10, 0.0])
    assert result["equity"].tolist() == pytest.approx([11_000.0, 11_000.0, 12_100.0, 12_100.0])


def test_run_backtest_deducts_commission_on_every_position_change() -> None:
    close = pd.Series([100.0, 110.0, 99.0, 108.9])
    signal = pd.Series([1, 0, 1, 0])  # position flips every day -> 4 trades, incl. entering on day 0

    result = run_backtest(close, signal, commission_bps=100.0, initial_capital=10_000.0)  # 1%

    # Last day has no next-day price to earn, so it has no return and no cost either.
    assert result["daily_return"].tolist() == pytest.approx([0.09, -0.01, 0.09, 0.0])


def test_run_backtest_no_commission_while_signal_never_changes() -> None:
    close = pd.Series([100.0, 105.0, 110.0])
    signal = pd.Series([1, 1, 1])

    result = run_backtest(close, signal, commission_bps=50.0)

    # Position only changes once, when entering on day 0.
    expected_first_day = 0.05 - 0.005  # 5% return minus the one-time 0.5% entry cost
    assert result["daily_return"].iloc[0] == pytest.approx(expected_first_day)
    assert result["daily_return"].iloc[1] == pytest.approx(110 / 105 - 1)


def test_compute_metrics_total_return_matches_compounded_returns() -> None:
    daily_return = pd.Series([0.10, 0.10])
    position = pd.Series([1, 1])

    metrics = compute_metrics(daily_return, position)

    expected_total_return = 1.10 * 1.10 - 1
    assert metrics["total_return"] == pytest.approx(expected_total_return)
    expected_cagr = (1 + expected_total_return) ** (252 / 2) - 1
    assert metrics["cagr"] == pytest.approx(expected_cagr)


def test_compute_metrics_max_drawdown_matches_hand_computed_path() -> None:
    daily_return = pd.Series([0.10, -0.20, 0.10, 0.10])
    position = pd.Series([1, 1, 1, 1])

    metrics = compute_metrics(daily_return, position)

    # Growth path: 1.10, 0.88, 0.968, 1.0648 -> peak is 1.10, trough is 0.88.
    expected_max_drawdown = 0.88 / 1.10 - 1
    assert metrics["max_drawdown"] == pytest.approx(expected_max_drawdown)


def test_compute_metrics_win_rate_only_counts_invested_days() -> None:
    # Flat day (position 0) has a nonzero return in the data but must not count.
    daily_return = pd.Series([0.05, -0.03, 0.02, -0.01])
    position = pd.Series([1, 1, 0, 1])

    metrics = compute_metrics(daily_return, position)

    # Invested days are 0.05 (win), -0.03 (loss), -0.01 (loss) -> 1/3.
    assert metrics["win_rate"] == pytest.approx(1 / 3)


def test_compute_metrics_sharpe_matches_hand_computed_value() -> None:
    daily_return = pd.Series([0.01, 0.02, 0.015])
    position = pd.Series([1, 1, 1])

    metrics = compute_metrics(daily_return, position)

    expected_sharpe = (daily_return.mean() / daily_return.std()) * math.sqrt(252)
    assert metrics["sharpe"] == pytest.approx(expected_sharpe)


def test_compute_metrics_sharpe_is_zero_when_returns_never_vary() -> None:
    daily_return = pd.Series([0.0, 0.0, 0.0])
    position = pd.Series([0, 0, 0])

    metrics = compute_metrics(daily_return, position)

    assert metrics["sharpe"] == 0.0
    assert metrics["win_rate"] == 0.0
