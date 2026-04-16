#!/usr/bin/env python3
"""Run a backtest using NautilusTrader BacktestEngine.

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --symbols AAPL MSFT GOOGL --years 3
"""

import argparse
import sys
import time
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yfinance as yf

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.model.data import Bar, BarSpecification, BarType
from nautilus_trader.model.enums import (
    AccountType,
    AggregationSource,
    BarAggregation,
    OmsType,
    PriceType,
)
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Currency, Money, Price, Quantity

from kryor.strategy.momentum import MomentumConfig, MomentumStrategy
from kryor.strategy.mean_reversion import MeanReversionConfig, MeanReversionStrategy

VENUE = Venue("ALPACA")
USD = Currency.from_str("USD")

DEFAULT_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
    "JPM", "V", "JNJ", "UNH", "HD", "PG",
    "SPY",
]


def create_instrument(symbol: str) -> Equity:
    ts = int(time.time() * 1e9)
    return Equity(
        instrument_id=InstrumentId(Symbol(symbol), VENUE),
        raw_symbol=Symbol(symbol),
        currency=USD,
        price_precision=2,
        price_increment=Price.from_str("0.01"),
        lot_size=Quantity.from_int(1),
        ts_event=ts,
        ts_init=ts,
        maker_fee=Decimal("0"),
        taker_fee=Decimal("0"),
    )


def fetch_bars(symbol: str, start: datetime, end: datetime) -> list[Bar]:
    instrument_id = InstrumentId(Symbol(symbol), VENUE)
    bar_spec = BarSpecification(1, BarAggregation.DAY, PriceType.LAST)
    bar_type = BarType(instrument_id, bar_spec, AggregationSource.EXTERNAL)

    df = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=True)
    bars = []
    for ts_idx, row in df.iterrows():
        ts_ns = int(ts_idx.timestamp() * 1e9)
        bar = Bar(
            bar_type=bar_type,
            open=Price.from_str(f"{row['Open']:.2f}"),
            high=Price.from_str(f"{row['High']:.2f}"),
            low=Price.from_str(f"{row['Low']:.2f}"),
            close=Price.from_str(f"{row['Close']:.2f}"),
            volume=Quantity.from_int(int(row["Volume"])),
            ts_event=ts_ns,
            ts_init=ts_ns,
        )
        bars.append(bar)
    return bars


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NautilusTrader backtest")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--years", type=int, default=3)
    parser.add_argument("--capital", type=float, default=50000)
    args = parser.parse_args()

    end = datetime.now()
    start = end - timedelta(days=args.years * 365)

    # ── Engine setup ──────────────────────────────────────

    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id="KRYOR-BT-001",
        ),
    )

    engine.add_venue(
        venue=VENUE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        starting_balances=[Money(args.capital, USD)],
    )

    # ── Load data ─────────────────────────────────────────

    all_bars = []
    for sym in args.symbols:
        print(f"Fetching {sym}...")
        instrument = create_instrument(sym)
        engine.add_instrument(instrument)
        bars = fetch_bars(sym, start, end)
        all_bars.extend(bars)
        print(f"  {len(bars)} bars")

    engine.add_data(all_bars)
    print(f"Total: {len(all_bars)} bars loaded")

    # ── Strategies ────────────────────────────────────────

    momentum = MomentumStrategy(
        config=MomentumConfig(
            strategy_id="MOM-BT",
            symbols=args.symbols,
            lookback_days=252,
            skip_days=21,
            top_pct=0.10,
            rebalance_interval_days=21,
        ),
    )
    engine.add_strategy(momentum)

    mean_rev = MeanReversionStrategy(
        config=MeanReversionConfig(
            strategy_id="MR-BT",
            symbols=args.symbols,
            rsi_threshold=30,
            volume_mult=1.5,
            max_hold_days=5,
        ),
    )
    engine.add_strategy(mean_rev)

    # ── Run ───────────────────────────────────────────────

    print("\nRunning backtest...")
    engine.run()

    # ── Results ───────────────────────────────────────────

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    # Account report
    for report in engine.trader.generate_account_report(VENUE):
        print(report)

    # Order fills
    fills = engine.trader.generate_order_fills_report()
    if fills is not None and not fills.empty:
        print(f"\nTotal fills: {len(fills)}")
        print(fills.tail(10))

    # Position report
    positions = engine.trader.generate_positions_report()
    if positions is not None and not positions.empty:
        print(f"\nPositions: {len(positions)}")
        print(positions)

    engine.dispose()
    print("\nBacktest complete.")


if __name__ == "__main__":
    main()
