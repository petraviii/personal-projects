"""Command-line entrypoints for papertrader."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from papertrader.backtest import compute_metrics, run_backtest
from papertrader.data_fetch import fetch_and_store
from papertrader.features import build_feature_matrix
from papertrader.labeling import make_labels
from papertrader.model import evaluate_walk_forward, walk_forward_predictions
from papertrader.storage import DEFAULT_DB_PATH, get_connection, read_ohlcv


def _load_close(ticker: str, db_path: Path) -> pd.Series:
    conn = get_connection(db_path)
    try:
        bars = read_ohlcv(conn, ticker)
    finally:
        conn.close()
    if bars.empty:
        raise ValueError(f"No cached data for '{ticker}'. Run `papertrader fetch` first.")
    return bars["close"]


def _fetch(args: argparse.Namespace) -> None:
    rows_written = fetch_and_store(
        tickers=args.tickers, start=args.start, end=args.end, db_path=args.db_path
    )
    for ticker, count in rows_written.items():
        print(f"{ticker}: {count} rows cached")


def _train(args: argparse.Namespace) -> None:
    close = _load_close(args.ticker, args.db_path)
    features = build_feature_matrix(close)
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


def _walk_forward_signal(args: argparse.Namespace) -> tuple[pd.Series, pd.Series]:
    """Shared by `backtest` and `report`: close prices plus a lookahead-free
    long/flat signal built from stitched walk-forward out-of-sample predictions."""
    close = _load_close(args.ticker, args.db_path)
    features = build_feature_matrix(close)
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


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ticker", required=True, help="e.g. AAPL")
    parser.add_argument("--horizon", type=int, default=5, help="label lookahead in days (default: 5)")
    parser.add_argument(
        "--threshold", type=float, default=0.0, help="forward-return cutoff for an up label (default: 0.0)"
    )
    parser.add_argument("--n-splits", type=int, default=5, help="walk-forward fold count (default: 5)")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)


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

    train_parser = subparsers.add_parser("train", help="Walk-forward evaluate the baseline classifier")
    _add_model_args(train_parser)
    train_parser.set_defaults(func=_train)

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

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    args.func(args)


if __name__ == "__main__":
    main()
