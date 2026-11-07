"""news.py tests: deterministic scoring/aggregation, plus a network smoke
test for fetch_and_store_sentiment (mirrors test_data_fetch.py's pattern)."""

from pathlib import Path

import pandas as pd

from papertrader.news import (
    daily_sentiment,
    fetch_and_store_sentiment,
    score_headline,
)
from papertrader.storage import get_connection, read_sentiment


def test_score_headline_is_positive_for_bullish_words() -> None:
    assert score_headline("Stock surges after strong earnings beat") > 0


def test_score_headline_is_negative_for_bearish_words() -> None:
    assert score_headline("Shares plunge on weak guidance and lawsuit") < 0


def test_score_headline_is_neutral_with_no_lexicon_words() -> None:
    assert score_headline("Company holds annual shareholder meeting") == 0.0


def test_score_headline_nets_mixed_sentiment() -> None:
    # One positive ("surge"), one negative ("warns") -> net zero.
    assert score_headline("Stock surges even as CEO warns of risks") == 0.0


def test_daily_sentiment_averages_same_day_headlines() -> None:
    headlines = [
        {"title": "Stock surges on record profit", "date": pd.Timestamp("2024-01-02", tz="UTC")},
        {"title": "Shares plunge on fraud lawsuit", "date": pd.Timestamp("2024-01-02", tz="UTC")},
        {"title": "Company beats on strong growth", "date": pd.Timestamp("2024-01-03", tz="UTC")},
    ]

    result = daily_sentiment(headlines)

    assert list(result.index.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-03"]
    assert result.loc["2024-01-02"] == 0.0  # one +1 and one -1 headline average to 0
    assert result.loc["2024-01-03"] == 1.0


def test_daily_sentiment_on_no_headlines_is_empty() -> None:
    result = daily_sentiment([])
    assert result.empty


def test_fetch_and_store_sentiment_populates_the_cache(tmp_path: Path) -> None:
    db_path = tmp_path / "news_smoke.sqlite"
    rows_written = fetch_and_store_sentiment(tickers=["AAPL"], db_path=db_path)

    # Yahoo's recent-news window can legitimately be empty; just check it
    # ran without error and the cache reflects whatever came back.
    assert rows_written["AAPL"] >= 0

    conn = get_connection(db_path)
    cached = read_sentiment(conn, "AAPL")
    assert len(cached) == rows_written["AAPL"]
