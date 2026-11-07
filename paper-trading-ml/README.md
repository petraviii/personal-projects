# papertrader

A paper trading engine that uses machine learning to generate buy/flat
signals on US equities, then simulates how a portfolio following those
signals would have performed.

> **Disclaimer:** This is an educational / portfolio project. It simulates
> trades against historical data (paper trading) — it does not place real
> orders and is not investment advice.

## Architecture

```mermaid
flowchart LR
    A[yfinance] -->|daily OHLCV| B[storage.py<br/>SQLite cache]
    B --> C[features.py<br/>SMA / EMA / RSI / MACD / returns]
    C --> D[labeling.py<br/>forward N-day return -> up/down]
    D --> E[model.py<br/>walk-forward train/predict]
    E --> F[backtest.py<br/>simulated portfolio + metrics]
    F --> G[cli.py<br/>fetch / train / backtest / report]
```

## Status

Full pipeline is in place: data fetch, feature engineering, labeling, a
walk-forward-evaluated baseline classifier, and a backtest engine, wired
up end-to-end through the CLI. Model comparison/tuning and a news-sentiment
feature are tracked as stretch-goal issues.

## Setup

```bash
cd Claude/paper-trading-ml
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

**1. Fetch and cache daily OHLCV bars:**

```bash
python -m papertrader fetch --tickers AAPL --start 2023-01-01
```

This populates `data/papertrader.sqlite` with daily OHLCV rows for the
given tickers.

**2. Walk-forward evaluate the baseline classifier** (per-fold accuracy /
precision / recall, chronological expanding-window splits — no lookahead):

```bash
python -m papertrader train --ticker AAPL --horizon 5 --n-splits 5
```

```
 fold  train_size  test_size  skipped  accuracy  precision  recall
    0         100         80    False     0.538      0.512   0.600
    ...

Mean accuracy: 0.52  Mean precision: 0.51  Mean recall: 0.55
```

**3. Simulate trading** on the stitched walk-forward out-of-sample
predictions (each day's signal only ever uses a model trained on data
strictly before it):

```bash
python -m papertrader backtest --ticker AAPL --commission-bps 5 --output equity.csv
```

Prints the tail of the simulated equity curve and optionally writes the
full curve (`position`, `daily_return`, `equity` per day) to a CSV.

**4. One-shot performance summary:**

```bash
python -m papertrader report --ticker AAPL
```

```
Ticker: AAPL
Period: 2023-01-03 to 2024-12-30
Total return:    xx.xx%
CAGR:            xx.xx%
Sharpe:            x.xx
Max drawdown:    -x.xx%
Win rate:        xx.xx%
Final equity:  xxxxx.xx
```

Shared flags across `train`/`backtest`/`report`: `--horizon` (label
lookahead in days, default 5), `--threshold` (forward-return cutoff for an
up label, default 0.0), `--n-splits` (walk-forward fold count, default 5),
`--db-path` (default `data/papertrader.sqlite`). `backtest`/`report` also
take `--position-size`, `--commission-bps`, and `--initial-capital`.

## Tests

```bash
pytest -q
```
