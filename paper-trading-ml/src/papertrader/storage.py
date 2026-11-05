"""SQLite-backed cache for daily OHLCV data."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

DEFAULT_DB_PATH = Path("data/papertrader.sqlite")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ohlcv (
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume INTEGER NOT NULL,
    PRIMARY KEY (ticker, date)
);
"""


def get_connection(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(_SCHEMA)
    conn.commit()
    return conn


def upsert_ohlcv(conn: sqlite3.Connection, ticker: str, bars: pd.DataFrame) -> int:
    """Insert or replace daily bars for `ticker`. `bars` is indexed by date
    with columns open/high/low/close/volume (case-insensitive)."""
    normalized = bars.rename(columns=str.lower)
    rows = [
        (
            ticker,
            index.strftime("%Y-%m-%d"),
            float(row["open"]),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
            int(row["volume"]),
        )
        for index, row in normalized.iterrows()
    ]
    conn.executemany(
        """
        INSERT OR REPLACE INTO ohlcv (ticker, date, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def read_ohlcv(conn: sqlite3.Connection, ticker: str) -> pd.DataFrame:
    """Return cached daily bars for `ticker`, indexed by date and sorted ascending."""
    df = pd.read_sql_query(
        "SELECT date, open, high, low, close, volume FROM ohlcv "
        "WHERE ticker = ? ORDER BY date ASC",
        conn,
        params=(ticker,),
        parse_dates=["date"],
    )
    return df.set_index("date")
