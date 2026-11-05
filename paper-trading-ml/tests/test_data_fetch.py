"""Smoke test: fetch_and_store should populate the SQLite cache from a live
Yahoo Finance request. Network-dependent by design (small, local verification)."""

from pathlib import Path

from papertrader.data_fetch import fetch_and_store
from papertrader.storage import get_connection, read_ohlcv


def test_fetch_and_store_populates_db(tmp_path: Path) -> None:
    db_path = tmp_path / "smoke.sqlite"
    rows_written = fetch_and_store(
        tickers=["AAPL"], start="2024-01-01", end="2024-01-15", db_path=db_path
    )

    assert rows_written["AAPL"] > 0

    conn = get_connection(db_path)
    cached = read_ohlcv(conn, "AAPL")
    assert len(cached) == rows_written["AAPL"]
