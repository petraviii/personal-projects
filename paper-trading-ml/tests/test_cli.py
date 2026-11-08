"""cli.py tests for the model-training/signal helper and the `run-daily`
automation loop -- no real network/Alpaca calls, so CI needs no credentials."""

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from papertrader import cli
from papertrader.storage import get_connection, upsert_ohlcv
from test_broker import FakeTradingClient


FIXED_TODAY = date(2024, 6, 12)  # an arbitrary Wednesday, so weekday/weekend timing never flakes the tests


class _FixedDate(date):
    """Stand-in for `datetime.date` so `date.today()` inside cli.py is deterministic."""

    @classmethod
    def today(cls) -> date:
        return FIXED_TODAY


def _seed_prices(db_path: Path, ticker: str, last_date: date, n_days: int = 400) -> None:
    """Cache `n_days` of a trending-with-noise close series ending on `last_date`,
    enough warmup history for the feature indicators to be fully computed."""
    index = pd.bdate_range(end=last_date, periods=n_days)
    rng = np.random.default_rng(42)
    closes = 100 + np.cumsum(rng.normal(0.05, 1.0, size=n_days))
    bars = pd.DataFrame(
        {
            "Open": closes,
            "High": closes + 1,
            "Low": closes - 1,
            "Close": closes,
            "Volume": 1_000_000,
        },
        index=index,
    )
    conn = get_connection(db_path)
    try:
        upsert_ohlcv(conn, ticker, bars)
    finally:
        conn.close()


def _model_args(ticker: str, db_path: Path) -> argparse.Namespace:
    return argparse.Namespace(ticker=ticker, db_path=db_path, use_sentiment=False, horizon=5, threshold=0.0)


def test_train_and_signal_returns_zero_or_one(tmp_path: Path) -> None:
    db_path = tmp_path / "test.sqlite"
    _seed_prices(db_path, "AAPL", last_date=date.today())
    close = cli._load_close("AAPL", db_path)

    signal = cli._train_and_signal(_model_args("AAPL", db_path), close)

    assert signal in (0, 1)


def test_train_and_signal_raises_when_not_enough_warmup_history(tmp_path: Path) -> None:
    db_path = tmp_path / "test.sqlite"
    _seed_prices(db_path, "AAPL", last_date=date.today(), n_days=10)
    close = cli._load_close("AAPL", db_path)

    with pytest.raises(ValueError, match="missing indicator"):
        cli._train_and_signal(_model_args("AAPL", db_path), close)


def _run_daily_args(tickers: list[str], db_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        tickers=tickers, db_path=db_path, use_sentiment=False, horizon=5, threshold=0.0, lookback_days=730
    )


def test_run_daily_submits_an_order_for_a_fresh_ticker(tmp_path, monkeypatch, capsys) -> None:
    db_path = tmp_path / "test.sqlite"
    _seed_prices(db_path, "AAPL", last_date=FIXED_TODAY)

    fake_client = FakeTradingClient()
    monkeypatch.setattr(cli, "date", _FixedDate)
    monkeypatch.setattr(cli, "build_client", lambda: fake_client)
    monkeypatch.setattr(cli, "fetch_and_store", lambda **kwargs: None)

    cli._run_daily(_run_daily_args(["AAPL"], db_path))

    assert "ERROR" not in capsys.readouterr().out


def test_run_daily_skips_a_ticker_with_no_new_data_today(tmp_path, monkeypatch, capsys) -> None:
    db_path = tmp_path / "test.sqlite"
    stale_date = FIXED_TODAY - timedelta(days=5)
    _seed_prices(db_path, "MSFT", last_date=stale_date)

    fake_client = FakeTradingClient()
    monkeypatch.setattr(cli, "date", _FixedDate)
    monkeypatch.setattr(cli, "build_client", lambda: fake_client)
    # Market closed / no new bars -- the cache still ends on `stale_date`.
    monkeypatch.setattr(cli, "fetch_and_store", lambda **kwargs: None)

    cli._run_daily(_run_daily_args(["MSFT"], db_path))

    assert "no new data for today" in capsys.readouterr().out
    assert fake_client.submitted_orders == []
    assert fake_client.closed_symbols == []


def test_run_daily_one_failing_ticker_does_not_stop_the_rest(tmp_path, monkeypatch, capsys) -> None:
    db_path = tmp_path / "test.sqlite"
    _seed_prices(db_path, "AAPL", last_date=FIXED_TODAY)
    _seed_prices(db_path, "MSFT", last_date=FIXED_TODAY)

    fake_client = FakeTradingClient()
    monkeypatch.setattr(cli, "date", _FixedDate)
    monkeypatch.setattr(cli, "build_client", lambda: fake_client)
    monkeypatch.setattr(cli, "fetch_and_store", lambda **kwargs: None)

    original_train_and_signal = cli._train_and_signal

    def flaky_train_and_signal(args, close):
        if args.ticker == "AAPL":
            raise RuntimeError("simulated model failure")
        return original_train_and_signal(args, close)

    monkeypatch.setattr(cli, "_train_and_signal", flaky_train_and_signal)

    cli._run_daily(_run_daily_args(["AAPL", "MSFT"], db_path))

    out = capsys.readouterr().out
    assert "AAPL: ERROR - simulated model failure" in out
    assert "MSFT: signal=" in out
