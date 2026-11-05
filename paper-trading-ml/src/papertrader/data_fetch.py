"""Fetch daily OHLCV bars from Yahoo Finance and cache them in SQLite."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from papertrader.storage import DEFAULT_DB_PATH, get_connection, upsert_ohlcv


def fetch_ticker(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Download daily OHLCV bars for a single ticker from Yahoo Finance."""
    bars = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if bars.empty:
        raise ValueError(f"No data returned for ticker '{ticker}' (start={start}, end={end})")
    if isinstance(bars.columns, pd.MultiIndex):
        # Recent yfinance versions return (field, ticker) columns even for a
        # single symbol; drop the ticker level to get plain field names.
        bars.columns = bars.columns.get_level_values(0)
    return bars[["Open", "High", "Low", "Close", "Volume"]]


def fetch_and_store(
    tickers: list[str],
    start: str,
    end: str | None = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> dict[str, int]:
    """Fetch each ticker and upsert into the SQLite cache. Returns rows written per ticker."""
    conn = get_connection(db_path)
    try:
        rows_written: dict[str, int] = {}
        for ticker in tickers:
            bars = fetch_ticker(ticker, start=start, end=end)
            rows_written[ticker] = upsert_ohlcv(conn, ticker, bars)
        return rows_written
    finally:
        conn.close()
