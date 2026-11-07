"""Optional news-sentiment feature: score cached headlines with a small
hand-rolled finance lexicon (kept dependency-free, same rationale as the
hand-rolled indicators in features.py) and aggregate to one score per day.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import yfinance as yf

from papertrader.storage import DEFAULT_DB_PATH, get_connection, upsert_sentiment

POSITIVE_WORDS = {
    "beat", "beats", "surge", "surges", "soar", "soars", "rally", "rallies",
    "upgrade", "upgrades", "upgraded", "profit", "profits", "growth", "record",
    "strong", "bullish", "gain", "gains", "outperform", "outperforms", "jump",
    "jumps", "boost", "boosts", "win", "wins", "breakthrough", "optimistic",
}

NEGATIVE_WORDS = {
    "miss", "misses", "plunge", "plunges", "slump", "slumps", "downgrade",
    "downgrades", "downgraded", "loss", "losses", "weak", "bearish", "decline",
    "declines", "lawsuit", "recall", "fraud", "crash", "crashes", "warn",
    "warns", "cut", "cuts", "layoff", "layoffs", "investigation", "pessimistic",
}

_WORD_RE = re.compile(r"[a-z']+")


def score_headline(title: str) -> float:
    """Net positive-minus-negative word count, normalized by the number of
    lexicon hits so scores stay roughly in [-1, 1]. A headline that matches
    no lexicon word scores neutral (0.0)."""
    words = _WORD_RE.findall(title.lower())
    positive_hits = sum(1 for w in words if w in POSITIVE_WORDS)
    negative_hits = sum(1 for w in words if w in NEGATIVE_WORDS)
    total_hits = positive_hits + negative_hits
    if total_hits == 0:
        return 0.0
    return (positive_hits - negative_hits) / total_hits


def fetch_headlines(ticker: str) -> list[dict]:
    """Recent headlines for `ticker` from Yahoo Finance, as {title, date}.

    Yahoo only exposes a short rolling window of recent news (not
    historical), so this can't backfill sentiment for past dates.
    """
    raw_items = yf.Ticker(ticker).news or []
    headlines = []
    for item in raw_items:
        content = item.get("content", item)
        title = content.get("title")
        pub_date = content.get("pubDate") or content.get("providerPublishTime")
        if not title or not pub_date:
            continue
        headlines.append({"title": title, "date": pd.to_datetime(pub_date, utc=True).normalize()})
    return headlines


def daily_sentiment(headlines: list[dict]) -> pd.Series:
    """Average headline sentiment score per day, indexed by date."""
    if not headlines:
        return pd.Series(dtype=float)
    scored = pd.DataFrame(headlines)
    scored["score"] = scored["title"].map(score_headline)
    return scored.groupby(scored["date"].dt.tz_localize(None))["score"].mean()


def fetch_and_store_sentiment(
    tickers: list[str], db_path: Path = DEFAULT_DB_PATH
) -> dict[str, int]:
    """Fetch each ticker's recent headlines, score them, and upsert the
    per-day average sentiment into the SQLite cache. Returns rows written
    per ticker (0 if Yahoo has no recent news for it)."""
    conn = get_connection(db_path)
    try:
        rows_written: dict[str, int] = {}
        for ticker in tickers:
            sentiment = daily_sentiment(fetch_headlines(ticker))
            rows_written[ticker] = upsert_sentiment(conn, ticker, sentiment)
        return rows_written
    finally:
        conn.close()
