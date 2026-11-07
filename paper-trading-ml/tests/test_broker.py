"""broker.py tests against a fake TradingClient stub — no real Alpaca
network calls, so CI needs no credentials."""

from types import SimpleNamespace

from papertrader.broker import get_account_summary, get_positions, submit_signal_order


class FakeOrder:
    def __init__(self, order_id: str = "order-1") -> None:
        self.id = order_id


class FakeTradingClient:
    """Mimics the handful of TradingClient methods broker.py calls."""

    def __init__(self, buying_power: float = 1000.0, positions: list | None = None) -> None:
        self._buying_power = buying_power
        self._positions = positions or []
        self.submitted_orders: list = []
        self.closed_symbols: list[str] = []

    def get_account(self) -> SimpleNamespace:
        return SimpleNamespace(cash=500.0, equity=1500.0, buying_power=self._buying_power)

    def get_all_positions(self) -> list:
        return self._positions

    def submit_order(self, order_data) -> FakeOrder:
        self.submitted_orders.append(order_data)
        return FakeOrder()

    def close_position(self, symbol: str) -> FakeOrder:
        self.closed_symbols.append(symbol)
        return FakeOrder()


def _fake_position(symbol: str, qty: float) -> SimpleNamespace:
    return SimpleNamespace(
        symbol=symbol, qty=qty, avg_entry_price=100.0, current_price=110.0, unrealized_pl=10.0
    )


def test_get_account_summary_reads_cash_equity_buying_power() -> None:
    client = FakeTradingClient(buying_power=2500.0)
    summary = get_account_summary(client)

    assert summary == {"cash": 500.0, "equity": 1500.0, "buying_power": 2500.0}


def test_get_positions_returns_a_dataframe_row_per_position() -> None:
    client = FakeTradingClient(positions=[_fake_position("AAPL", 3.0)])
    positions = get_positions(client)

    assert len(positions) == 1
    assert positions.iloc[0]["symbol"] == "AAPL"
    assert positions.iloc[0]["qty"] == 3.0


def test_submit_signal_order_buys_when_flat_and_signal_is_one() -> None:
    client = FakeTradingClient(buying_power=1000.0, positions=[])
    result = submit_signal_order(client, "AAPL", signal=1)

    assert result["action"] == "buy"
    assert len(client.submitted_orders) == 1
    assert client.submitted_orders[0].notional == 1000.0


def test_submit_signal_order_is_a_noop_when_already_holding_and_signal_is_one() -> None:
    client = FakeTradingClient(positions=[_fake_position("AAPL", 2.0)])
    result = submit_signal_order(client, "AAPL", signal=1)

    assert result is None
    assert client.submitted_orders == []


def test_submit_signal_order_closes_when_holding_and_signal_is_zero() -> None:
    client = FakeTradingClient(positions=[_fake_position("AAPL", 2.0)])
    result = submit_signal_order(client, "AAPL", signal=0)

    assert result == {"action": "close", "symbol": "AAPL", "qty": 2.0, "order_id": "order-1"}
    assert client.closed_symbols == ["AAPL"]


def test_submit_signal_order_is_a_noop_when_flat_and_signal_is_zero() -> None:
    client = FakeTradingClient(positions=[])
    result = submit_signal_order(client, "AAPL", signal=0)

    assert result is None
    assert client.closed_symbols == []
