"""Command-line entrypoints for papertrader."""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from papertrader.backtest import compute_metrics, run_backtest
from papertrader.broker import build_client, submit_signal_order
from papertrader.data_fetch import fetch_and_store
from papertrader.features import build_feature_matrix
from papertrader.labeling import make_labels
from papertrader.model import (
    evaluate_walk_forward,
    grid_search_walk_forward,
    random_forest_factory,
    train_final_model,
    walk_forward_predictions,
)
from papertrader.news import fetch_and_store_sentiment
from papertrader.storage import DEFAULT_DB_PATH, get_connection, read_ohlcv, read_sentiment


def _load_close(ticker: str, db_path: Path) -> pd.Series:
    conn = get_connection(db_path)
    try:
        bars = read_ohlcv(conn, ticker)
    finally:
        conn.close()
    if bars.empty:
        raise ValueError(f"No cached data for '{ticker}'. Run `papertrader fetch` first.")
    return bars["close"]


def _build_features(args: argparse.Namespace, close: pd.Series) -> pd.DataFrame:
    if not args.use_sentiment:
        return build_feature_matrix(close)
    conn = get_connection(args.db_path)
    try:
        sentiment = read_sentiment(conn, args.ticker)
    finally:
        conn.close()
    return build_feature_matrix(close, sentiment=sentiment)


def _fetch(args: argparse.Namespace) -> None:
    rows_written = fetch_and_store(
        tickers=args.tickers, start=args.start, end=args.end, db_path=args.db_path
    )
    for ticker, count in rows_written.items():
        print(f"{ticker}: {count} rows cached")


def _fetch_news(args: argparse.Namespace) -> None:
    rows_written = fetch_and_store_sentiment(tickers=args.tickers, db_path=args.db_path)
    for ticker, count in rows_written.items():
        print(f"{ticker}: {count} days of sentiment cached")


def _train(args: argparse.Namespace) -> None:
    close = _load_close(args.ticker, args.db_path)
    features = _build_features(args, close)
    labels = make_labels(close, horizon=args.horizon, threshold=args.threshold)

    results = evaluate_walk_forward(features, labels, n_splits=args.n_splits)
    print(results.to_string(index=False))

    evaluated = results[~results["skipped"]]
    if evaluated.empty:
        print("\nNo fold had both classes in training; nothing was evaluated.")
        return
    print(
        f"\nMean accuracy: {evaluated['accuracy'].mean():.3f}  "
        f"Mean precision: {evaluated['precision'].mean():.3f}  "
        f"Mean recall: {evaluated['recall'].mean():.3f}"
    )


def _tune(args: argparse.Namespace) -> None:
    close = _load_close(args.ticker, args.db_path)
    features = _build_features(args, close)
    labels = make_labels(close, horizon=args.horizon, threshold=args.threshold)

    param_grid = {
        "n_estimators": args.n_estimators,
        "max_depth": [None if d == 0 else d for d in args.max_depth],
    }
    grid = grid_search_walk_forward(
        features, labels, random_forest_factory, param_grid, n_splits=args.n_splits
    )
    print(grid.to_string(index=False))

    baseline = evaluate_walk_forward(features, labels, n_splits=args.n_splits)
    baseline_evaluated = baseline[~baseline["skipped"]]
    baseline_accuracy = baseline_evaluated["accuracy"].mean() if len(baseline_evaluated) else float("nan")
    best_accuracy = grid["mean_accuracy"].iloc[0] if len(grid) else float("nan")

    print(f"\nLogisticRegression baseline mean accuracy: {baseline_accuracy:.3f}")
    print(f"Best RandomForest mean accuracy:            {best_accuracy:.3f}")


def _walk_forward_signal(args: argparse.Namespace) -> tuple[pd.Series, pd.Series]:
    """Shared by `backtest` and `report`: close prices plus a lookahead-free
    long/flat signal built from stitched walk-forward out-of-sample predictions."""
    close = _load_close(args.ticker, args.db_path)
    features = _build_features(args, close)
    labels = make_labels(close, horizon=args.horizon, threshold=args.threshold)

    predictions = walk_forward_predictions(features, labels, n_splits=args.n_splits)
    signal = predictions.reindex(close.index).fillna(0.0)
    return close, signal


def _backtest(args: argparse.Namespace) -> None:
    close, signal = _walk_forward_signal(args)
    result = run_backtest(
        close,
        signal,
        position_size=args.position_size,
        commission_bps=args.commission_bps,
        initial_capital=args.initial_capital,
    )

    print(result.tail(10).to_string())
    print(f"\nFinal equity: {result['equity'].iloc[-1]:.2f}")

    if args.output is not None:
        result.to_csv(args.output)
        print(f"Equity curve written to {args.output}")


def _report(args: argparse.Namespace) -> None:
    close, signal = _walk_forward_signal(args)
    result = run_backtest(
        close,
        signal,
        position_size=args.position_size,
        commission_bps=args.commission_bps,
        initial_capital=args.initial_capital,
    )
    metrics = compute_metrics(result["daily_return"], result["position"])

    print(f"Ticker: {args.ticker}")
    print(f"Period: {close.index.min().date()} to {close.index.max().date()}")
    print(f"Total return:  {metrics['total_return']:>8.2%}")
    print(f"CAGR:          {metrics['cagr']:>8.2%}")
    print(f"Sharpe:        {metrics['sharpe']:>8.2f}")
    print(f"Max drawdown:  {metrics['max_drawdown']:>8.2%}")
    print(f"Win rate:      {metrics['win_rate']:>8.2%}")
    print(f"Final equity:  {result['equity'].iloc[-1]:>8.2f}")


def _train_and_signal(args: argparse.Namespace, close: pd.Series) -> int:
    """Train the final model on all cached history and return today's signal
    (1=long, 0=flat) from the latest cached feature row."""
    features = _build_features(args, close)

    # Checked before training: with too little total history, an indicator
    # like sma_50 is NaN for every row including the last one, and
    # `train_final_model` would otherwise fail on an empty dropna'd dataset
    # with a cryptic sklearn error instead of this actionable message.
    latest_date = features.index[-1]
    latest_features = features.iloc[[-1]]
    if latest_features.isna().any().any():
        raise ValueError(
            f"Latest cached row for '{args.ticker}' ({latest_date.date()}) has missing indicator "
            "values (not enough warmup history yet) -- can't compute today's signal."
        )

    labels = make_labels(close, horizon=args.horizon, threshold=args.threshold)
    model = train_final_model(features, labels)
    return int(model.predict(latest_features)[0])


def _trade(args: argparse.Namespace) -> None:
    close = _load_close(args.ticker, args.db_path)
    signal = _train_and_signal(args, close)

    client = build_client()
    result = submit_signal_order(client, args.ticker, signal)

    print(f"Signal for {args.ticker} on {close.index[-1].date()}: {'long' if signal else 'flat'}")
    print("No order submitted (already in the right position)." if result is None else result)


def _run_daily(args: argparse.Namespace) -> None:
    """Fetch fresh data and submit today's signal for each ticker, meant to be
    called from cron. Skips a ticker cleanly if the market was closed (no new
    cached row for today); isolates failures per-ticker so one bad ticker
    doesn't stop the rest of an unattended run."""
    today = date.today()
    client = build_client()
    for ticker in args.tickers:
        try:
            start = (today - timedelta(days=args.lookback_days)).isoformat()
            fetch_and_store(tickers=[ticker], start=start, db_path=args.db_path)
            if args.use_sentiment:
                fetch_and_store_sentiment(tickers=[ticker], db_path=args.db_path)

            close = _load_close(ticker, args.db_path)
            latest_date = close.index[-1].date()
            if latest_date != today:
                print(f"{ticker}: no new data for today ({latest_date}), skipping")
                continue

            ticker_args = argparse.Namespace(
                ticker=ticker,
                db_path=args.db_path,
                use_sentiment=args.use_sentiment,
                horizon=args.horizon,
                threshold=args.threshold,
            )
            signal = _train_and_signal(ticker_args, close)
            result = submit_signal_order(client, ticker, signal)
            print(
                f"{ticker}: signal={'long' if signal else 'flat'} on {latest_date} -> "
                f"{result if result else 'no order (already correct position)'}"
            )
        except Exception as exc:
            print(f"{ticker}: ERROR - {exc}")


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ticker", required=True, help="e.g. AAPL")
    parser.add_argument("--horizon", type=int, default=5, help="label lookahead in days (default: 5)")
    parser.add_argument(
        "--threshold", type=float, default=0.0, help="forward-return cutoff for an up label (default: 0.0)"
    )
    parser.add_argument("--n-splits", type=int, default=5, help="walk-forward fold count (default: 5)")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--use-sentiment",
        action="store_true",
        help="add the cached news-sentiment feature (run `fetch-news` first)",
    )


def _add_backtest_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--position-size", type=float, default=1.0, help="fraction of capital per trade (default: 1.0)")
    parser.add_argument("--commission-bps", type=float, default=0.0, help="cost per position change, in bps (default: 0)")
    parser.add_argument("--initial-capital", type=float, default=10_000.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="papertrader")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch", help="Fetch and cache daily OHLCV bars")
    fetch_parser.add_argument("--tickers", nargs="+", required=True, help="e.g. AAPL MSFT")
    fetch_parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    fetch_parser.add_argument("--end", default=None, help="YYYY-MM-DD (default: today)")
    fetch_parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    fetch_parser.set_defaults(func=_fetch)

    fetch_news_parser = subparsers.add_parser(
        "fetch-news", help="Fetch and score recent headlines into the news-sentiment cache"
    )
    fetch_news_parser.add_argument("--tickers", nargs="+", required=True, help="e.g. AAPL MSFT")
    fetch_news_parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    fetch_news_parser.set_defaults(func=_fetch_news)

    train_parser = subparsers.add_parser("train", help="Walk-forward evaluate the baseline classifier")
    _add_model_args(train_parser)
    train_parser.set_defaults(func=_train)

    tune_parser = subparsers.add_parser(
        "tune", help="Grid-search RandomForest hyperparameters against the LogisticRegression baseline"
    )
    _add_model_args(tune_parser)
    tune_parser.add_argument(
        "--n-estimators", type=int, nargs="+", default=[100, 200], help="RandomForest tree counts to try"
    )
    tune_parser.add_argument(
        "--max-depth",
        type=int,
        nargs="+",
        default=[0, 5, 10],
        help="RandomForest max depths to try (0 means unlimited)",
    )
    tune_parser.set_defaults(func=_tune)

    backtest_parser = subparsers.add_parser(
        "backtest", help="Simulate trading on walk-forward out-of-sample predictions"
    )
    _add_model_args(backtest_parser)
    _add_backtest_args(backtest_parser)
    backtest_parser.add_argument("--output", type=Path, default=None, help="write the full equity curve to this CSV")
    backtest_parser.set_defaults(func=_backtest)

    report_parser = subparsers.add_parser("report", help="Print a one-shot performance summary")
    _add_model_args(report_parser)
    _add_backtest_args(report_parser)
    report_parser.set_defaults(func=_report)

    trade_parser = subparsers.add_parser(
        "trade", help="Submit today's model signal as a paper order via Alpaca"
    )
    _add_model_args(trade_parser)
    trade_parser.set_defaults(func=_trade)

    run_daily_parser = subparsers.add_parser(
        "run-daily",
        help="Fetch fresh data and submit today's signal for one or more tickers (for cron)",
    )
    run_daily_parser.add_argument("--tickers", nargs="+", required=True, help="e.g. AAPL MSFT")
    run_daily_parser.add_argument("--horizon", type=int, default=5, help="label lookahead in days (default: 5)")
    run_daily_parser.add_argument(
        "--threshold", type=float, default=0.0, help="forward-return cutoff for an up label (default: 0.0)"
    )
    run_daily_parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    run_daily_parser.add_argument(
        "--use-sentiment",
        action="store_true",
        help="add the cached news-sentiment feature (run `fetch-news` first)",
    )
    run_daily_parser.add_argument(
        "--lookback-days", type=int, default=730, help="days of history to keep cached per ticker (default: 730)"
    )
    run_daily_parser.set_defaults(func=_run_daily)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    args.func(args)


if __name__ == "__main__":
    main()
