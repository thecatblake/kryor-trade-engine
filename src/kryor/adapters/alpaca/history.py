"""Fetch historical bars from Alpaca REST API and convert to NT Bar objects.

Shared utility used by strategies to preload bar history on startup.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from nautilus_trader.model.data import Bar, BarSpecification, BarType
from nautilus_trader.model.enums import AggregationSource, BarAggregation, PriceType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.objects import Price, Quantity

from kryor.adapters.alpaca.constants import ALPACA_VENUE


def fetch_historical_bars(
    api_key: str,
    secret_key: str,
    symbol: str,
    days: int = 400,
) -> list[Bar]:
    """Fetch daily bars from Alpaca for a single symbol.

    Returns NT Bar objects sorted by timestamp ascending.
    """
    client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)

    start = datetime.now() - timedelta(days=days)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start,
    )

    try:
        result = client.get_stock_bars(req)
    except Exception:
        return []

    if symbol not in result.data:
        return []

    instrument_id = InstrumentId(Symbol(symbol), ALPACA_VENUE)
    bar_spec = BarSpecification(1, BarAggregation.DAY, PriceType.LAST)
    bar_type = BarType(instrument_id, bar_spec, AggregationSource.EXTERNAL)

    bars = []
    for ab in result.data[symbol]:
        ts = int(ab.timestamp.timestamp() * 1e9)
        bars.append(Bar(
            bar_type=bar_type,
            open=Price.from_str(f"{ab.open:.2f}"),
            high=Price.from_str(f"{ab.high:.2f}"),
            low=Price.from_str(f"{ab.low:.2f}"),
            close=Price.from_str(f"{ab.close:.2f}"),
            volume=Quantity.from_int(int(ab.volume)),
            ts_event=ts,
            ts_init=ts,
        ))
    return bars
