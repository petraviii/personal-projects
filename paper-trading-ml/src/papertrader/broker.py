"""Alpaca paper-trading integration: account/position/order reads, and
submitting today's long/flat signal as a market order.

Every function takes a `TradingClient` as a parameter (dependency
injection) rather than constructing one internally, so tests can pass a
fake client instead of hitting the real Alpaca API.
"""

from __future__ import annotations

import os

import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest


def build_client() -> TradingClient:
    """Build a paper-trading client from ALPACA_API_KEY_ID/ALPACA_SECRET_KEY."""
    api_key = os.environ.get("ALPACA_API_KEY_ID")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError(
            "Set ALPACA_API_KEY_ID and ALPACA_SECRET_KEY (see README) to use the Alpaca integration."
        )
    return TradingClient(api_key, secret_key, paper=True)


def get_account_summary(client: TradingClient) -> dict[str, float]:
    account = client.get_account()
    return {
        "cash": float(account.cash),
        "equity": float(account.equity),
        "buying_power": float(account.buying_power),
    }


def get_positions(client: TradingClient) -> pd.DataFrame:
    positions = client.get_all_positions()
    return pd.DataFrame(
        [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "unrealized_pl": float(p.unrealized_pl),
            }
            for p in positions
        ]
    )


def get_recent_orders(client: TradingClient, limit: int = 10) -> pd.DataFrame:
    orders = client.get_orders()[:limit]
    return pd.DataFrame(
        [
            {
                "symbol": o.symbol,
                "side": o.side.value if hasattr(o.side, "value") else o.side,
                "qty": o.qty,
                "status": o.status.value if hasattr(o.status, "value") else o.status,
                "submitted_at": o.submitted_at,
            }
            for o in orders
        ]
    )


def _held_qty(client: TradingClient, ticker: str) -> float:
    for position in client.get_all_positions():
        if position.symbol == ticker:
            return float(position.qty)
    return 0.0


def submit_signal_order(client: TradingClient, ticker: str, signal: int) -> dict | None:
    """Act on today's signal: buy (using all available buying power) if
    `signal == 1` and not already holding `ticker`; close the position if
    `signal == 0` and currently holding it. No-op (returns None) otherwise.
    """
    held_qty = _held_qty(client, ticker)

    if signal == 1 and held_qty == 0:
        account = client.get_account()
        notional_amount = float(account.buying_power)
        # Submitting by notional dollar amount avoids needing a live quote just to size qty.
        order = client.submit_order(
            MarketOrderRequest(
                symbol=ticker,
                notional=round(notional_amount, 2),
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
        )
        return {"action": "buy", "symbol": ticker, "notional": notional_amount, "order_id": str(order.id)}

    if signal == 0 and held_qty != 0:
        order = client.close_position(ticker)
        return {"action": "close", "symbol": ticker, "qty": held_qty, "order_id": str(order.id)}

    return None
