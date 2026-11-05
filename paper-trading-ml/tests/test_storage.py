"""storage.py round-trip tests: what goes in via upsert should come back via read."""

from pathlib import Path

import pandas as pd

from papertrader.storage import get_connection, read_ohlcv, upsert_ohlcv


def _sample_bars() -> pd.DataFrame:
    # Two trading days of made-up OHLCV data for a fictional ticker.
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Volume": [1000, 1200],
        },
        index=index,
    )


def test_upsert_then_read_round_trip(tmp_path: Path) -> None:
    conn = get_connection(tmp_path / "test.sqlite")
    rows_written = upsert_ohlcv(conn, "TEST", _sample_bars())

    assert rows_written == 2

    result = read_ohlcv(conn, "TEST")
    assert list(result.index.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-03"]
    assert result.loc["2024-01-03", "close"] == 102.0


def test_upsert_is_idempotent_on_same_dates(tmp_path: Path) -> None:
    conn = get_connection(tmp_path / "test.sqlite")
    upsert_ohlcv(conn, "TEST", _sample_bars())
    # Re-inserting the same dates should replace, not duplicate, rows.
    upsert_ohlcv(conn, "TEST", _sample_bars())

    result = read_ohlcv(conn, "TEST")
    assert len(result) == 2


def test_read_missing_ticker_returns_empty(tmp_path: Path) -> None:
    conn = get_connection(tmp_path / "test.sqlite")
    result = read_ohlcv(conn, "NOPE")
    assert result.empty
