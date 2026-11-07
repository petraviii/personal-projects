"""CLI smoke tests: a deterministic synthetic price series seeded directly
into the SQLite cache (no network calls), run through each subcommand."""

from pathlib import Path

import pandas as pd
import pytest

from papertrader.cli import main
from papertrader.storage import get_connection, upsert_ohlcv


def _triangle_wave(n: int, period: int = 10, amplitude: float = 5.0, base: float = 100.0) -> list[float]:
    """Repeating up-5/down-5 price path, so forward returns alternate sign
    predictably — enough signal for the baseline classifier to learn from."""
    half = period // 2
    prices = []
    for i in range(n):
        pos = i % period
        if pos < half:
            prices.append(base + pos * (amplitude / half))
        else:
            prices.append(base + amplitude - (pos - half) * (amplitude / half))
    return prices


@pytest.fixture
def seeded_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "cli_test.sqlite"
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
    return db_path


def test_train_prints_fold_results(seeded_db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["train", "--ticker", "TEST", "--db-path", str(seeded_db), "--n-splits", "3"])
    out = capsys.readouterr().out

    assert "accuracy" in out
    assert "Mean accuracy" in out


def test_backtest_prints_final_equity_and_writes_csv(
    seeded_db: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_path = tmp_path / "equity.csv"
    main(
        [
            "backtest",
            "--ticker", "TEST",
            "--db-path", str(seeded_db),
            "--n-splits", "3",
            "--output", str(output_path),
        ]
    )
    out = capsys.readouterr().out

    assert "Final equity" in out
    assert output_path.exists()
    written = pd.read_csv(output_path)
    assert {"position", "daily_return", "equity"}.issubset(written.columns)


def test_report_prints_summary_metrics(seeded_db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["report", "--ticker", "TEST", "--db-path", str(seeded_db), "--n-splits", "3"])
    out = capsys.readouterr().out

    assert "Sharpe" in out
    assert "Final equity" in out


def test_train_raises_a_clear_error_for_an_uncached_ticker(seeded_db: Path) -> None:
    with pytest.raises(ValueError, match="No cached data"):
        main(["train", "--ticker", "NOPE", "--db-path", str(seeded_db), "--n-splits", "3"])


def test_tune_prints_grid_and_baseline_comparison(
    seeded_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(
        [
            "tune",
            "--ticker", "TEST",
            "--db-path", str(seeded_db),
            "--n-splits", "3",
            "--n-estimators", "10", "20",
            "--max-depth", "0", "3",
        ]
    )
    out = capsys.readouterr().out

    assert "mean_accuracy" in out
    assert "LogisticRegression baseline mean accuracy" in out
    assert "Best RandomForest mean accuracy" in out
