"""Command-line entrypoints for papertrader."""

from __future__ import annotations

import argparse
import sys

from papertrader.data_fetch import fetch_and_store


def _fetch(args: argparse.Namespace) -> None:
    rows_written = fetch_and_store(tickers=args.tickers, start=args.start, end=args.end)
    for ticker, count in rows_written.items():
        print(f"{ticker}: {count} rows cached")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="papertrader")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch", help="Fetch and cache daily OHLCV bars")
    fetch_parser.add_argument("--tickers", nargs="+", required=True, help="e.g. AAPL MSFT")
    fetch_parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    fetch_parser.add_argument("--end", default=None, help="YYYY-MM-DD (default: today)")
    fetch_parser.set_defaults(func=_fetch)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    args.func(args)


if __name__ == "__main__":
    main()
