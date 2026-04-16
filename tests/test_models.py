"""Tests for core models and custom data types."""

import time

from kryor.core.custom_data import CircuitBreakerData, RegimeData, RegimeEnum


def test_regime_data_bull():
    ts = int(time.time() * 1e9)
    data = RegimeData(
        regime=RegimeEnum.BULL,
        probability=0.85,
        bull_prob=0.85,
        neutral_prob=0.10,
        bear_prob=0.05,
        ts_event=ts,
        ts_init=ts,
    )
    assert data.is_bull
    assert not data.is_bear
    assert data.kelly_fraction == 0.50


def test_regime_data_bear():
    ts = int(time.time() * 1e9)
    data = RegimeData(
        regime=RegimeEnum.BEAR,
        probability=0.80,
        bull_prob=0.05,
        neutral_prob=0.15,
        bear_prob=0.80,
        ts_event=ts,
        ts_init=ts,
    )
    assert data.is_bear
    assert data.kelly_fraction == 0.0


def test_regime_data_neutral_kelly():
    ts = int(time.time() * 1e9)
    data = RegimeData(
        regime=RegimeEnum.NEUTRAL,
        probability=0.60,
        bull_prob=0.20,
        neutral_prob=0.60,
        bear_prob=0.20,
        ts_event=ts,
        ts_init=ts,
    )
    assert data.kelly_fraction == 0.25


def test_circuit_breaker_data():
    ts = int(time.time() * 1e9)
    data = CircuitBreakerData(level=2, reason="Daily loss limit", ts_event=ts, ts_init=ts)
    assert data.level == 2
    assert data.reason == "Daily loss limit"


def test_alpaca_instrument_creation():
    from decimal import Decimal
    from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
    from nautilus_trader.model.instruments import Equity
    from nautilus_trader.model.objects import Currency, Price, Quantity

    ts = int(time.time() * 1e9)
    instrument_id = InstrumentId(Symbol("AAPL"), Venue("ALPACA"))
    equity = Equity(
        instrument_id=instrument_id,
        raw_symbol=Symbol("AAPL"),
        currency=Currency.from_str("USD"),
        price_precision=2,
        price_increment=Price.from_str("0.01"),
        lot_size=Quantity.from_int(1),
        ts_event=ts,
        ts_init=ts,
        maker_fee=Decimal("0"),
        taker_fee=Decimal("0"),
    )
    assert equity.id == instrument_id
    assert str(equity.quote_currency) == "USD"


def test_bar_creation():
    from nautilus_trader.model.data import Bar, BarSpecification, BarType
    from nautilus_trader.model.enums import AggregationSource, BarAggregation, PriceType
    from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
    from nautilus_trader.model.objects import Price, Quantity

    ts = int(time.time() * 1e9)
    instrument_id = InstrumentId(Symbol("AAPL"), Venue("ALPACA"))
    bar_spec = BarSpecification(1, BarAggregation.DAY, PriceType.LAST)
    bar_type = BarType(instrument_id, bar_spec, AggregationSource.EXTERNAL)

    bar = Bar(
        bar_type=bar_type,
        open=Price.from_str("150.00"),
        high=Price.from_str("155.00"),
        low=Price.from_str("149.00"),
        close=Price.from_str("153.00"),
        volume=Quantity.from_int(1000000),
        ts_event=ts,
        ts_init=ts,
    )
    assert float(bar.close) == 153.00
    assert float(bar.volume) == 1000000
