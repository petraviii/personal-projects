"""Smoke test for dashboard.py: importing it (its module-level script body)
should run end-to-end without raising, against a seeded in-memory-ish
SQLite cache -- no real Alpaca credentials, no network calls.

Streamlit's st.* calls log a harmless "missing ScriptRunContext" warning
when run outside `streamlit run` (as here); we only assert nothing raises.
All the real logic is covered by the already-tested library modules this
file just wires together.
"""

import importlib
from pathlib import Path

import pandas as pd
import pytest

from papertrader.storage import get_connection, upsert_ohlcv


def _triangle_wave(n: int, period: int = 10, amplitude: float = 5.0, base: float = 100.0) -> list[float]:
    half = period // 2
    prices = []
    for i in range(n):
        pos = i % period
        if pos < half:
            prices.append(base + pos * (amplitude / half))
        else:
            prices.append(base + amplitude - (pos - half) * (amplitude / half))
    return prices


def test_dashboard_runs_end_to_end_without_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "dashboard_test.sqlite"
    close = _triangle_wave(150)
    dates = pd.bdate_range("2023-01-02", periods=150)
    bars = pd.DataFrame(
        {
            "Open": close,
            "High": [c + 1 for c in close],
            "Low": [c - 1 for c in close],
            "Close": close,
            "Volume": [1_000] * len(close),
        },
        index=dates,
    )
    conn = get_connection(db_path)
    upsert_ohlcv(conn, "TEST", bars)
    conn.close()

    monkeypatch.setenv("PAPERTRADER_DB_PATH", str(db_path))
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)

    import papertrader.dashboard as dashboard

    importlib.reload(dashboard)
