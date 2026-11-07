"""Live dashboard: pick a cached ticker and see price/signal, equity curve,
drawdown, walk-forward fold accuracy, and summary metrics -- plus a live
Alpaca paper-account section when credentials are set.

Kept thin on purpose: every number here comes from an already-tested
library function (features/labeling/model/backtest/broker); this file is
just wiring + display calls.

Run with: streamlit run src/papertrader/dashboard.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from papertrader.backtest import compute_metrics, run_backtest
from papertrader.broker import build_client, get_account_summary, get_positions, get_recent_orders
from papertrader.features import build_feature_matrix
from papertrader.labeling import make_labels
from papertrader.model import evaluate_walk_forward, walk_forward_predictions
from papertrader.storage import DEFAULT_DB_PATH, get_connection, list_tickers, read_ohlcv, read_sentiment

st.set_page_config(page_title="papertrader", layout="wide")
st.title("papertrader dashboard")

db_path = Path(os.environ.get("PAPERTRADER_DB_PATH", str(DEFAULT_DB_PATH)))
conn = get_connection(db_path)
tickers = list_tickers(conn)

if not tickers:
    st.warning("No cached tickers yet. Run `papertrader fetch --tickers ... --start ...` first.")
    st.stop()

with st.sidebar:
    ticker = st.selectbox("Ticker", tickers)
    horizon = st.number_input("Label horizon (days)", min_value=1, value=5)
    threshold = st.number_input("Up-label threshold", value=0.0, step=0.01, format="%.3f")
    n_splits = st.number_input("Walk-forward folds", min_value=2, value=5)
    commission_bps = st.number_input("Commission (bps)", min_value=0.0, value=0.0)
    use_sentiment = st.checkbox("Use news sentiment", value=False)

close = read_ohlcv(conn, ticker)["close"]
sentiment = read_sentiment(conn, ticker) if use_sentiment else None
features = build_feature_matrix(close, sentiment=sentiment)
labels = make_labels(close, horizon=int(horizon), threshold=float(threshold))

predictions = walk_forward_predictions(features, labels, n_splits=int(n_splits))
signal = predictions.reindex(close.index).fillna(0.0)
result = run_backtest(close, signal, commission_bps=float(commission_bps))
metrics = compute_metrics(result["daily_return"], result["position"])

st.subheader(f"{ticker}: price & signal")
price_and_signal = pd.DataFrame({"close": close, "long (signal=1)": close.where(signal == 1)})
st.line_chart(price_and_signal)

metric_columns = st.columns(5)
metric_columns[0].metric("Total return", f"{metrics['total_return']:.2%}")
metric_columns[1].metric("CAGR", f"{metrics['cagr']:.2%}")
metric_columns[2].metric("Sharpe", f"{metrics['sharpe']:.2f}")
metric_columns[3].metric("Max drawdown", f"{metrics['max_drawdown']:.2%}")
metric_columns[4].metric("Win rate", f"{metrics['win_rate']:.2%}")

st.subheader("Equity curve")
st.line_chart(result["equity"])

st.subheader("Drawdown")
growth = (1 + result["daily_return"]).cumprod()
st.area_chart(growth / growth.cummax() - 1)

st.subheader("Walk-forward fold accuracy")
fold_results = evaluate_walk_forward(features, labels, n_splits=int(n_splits))
evaluated_folds = fold_results[~fold_results["skipped"]]
if evaluated_folds.empty:
    st.info("No fold had both classes in training; nothing to show.")
else:
    st.bar_chart(evaluated_folds.set_index("fold")["accuracy"])
st.dataframe(fold_results, hide_index=True)

st.subheader("Live Alpaca paper account")
if not (os.environ.get("ALPACA_API_KEY_ID") and os.environ.get("ALPACA_SECRET_KEY")):
    st.info("Set ALPACA_API_KEY_ID / ALPACA_SECRET_KEY to show your live paper account here (see README).")
else:

    @st.cache_data(ttl=60)
    def _load_account() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
        client = build_client()
        return get_account_summary(client), get_positions(client), get_recent_orders(client)

    if st.button("Refresh account"):
        _load_account.clear()

    account_summary, positions, orders = _load_account()
    account_columns = st.columns(3)
    account_columns[0].metric("Cash", f"${account_summary['cash']:,.2f}")
    account_columns[1].metric("Equity", f"${account_summary['equity']:,.2f}")
    account_columns[2].metric("Buying power", f"${account_summary['buying_power']:,.2f}")

    st.write("Positions")
    st.dataframe(positions, hide_index=True)
    st.write("Recent orders")
    st.dataframe(orders, hide_index=True)
