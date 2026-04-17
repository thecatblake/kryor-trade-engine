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

from kryor.regime.hmm_backtest import RegimeBacktestActor, RegimeBacktestConfig
from kryor.risk.circuit_breaker import CircuitBreakerActor, CircuitBreakerConfig
from kryor.strategy.momentum import MomentumConfig, MomentumStrategy
from kryor.strategy.mean_reversion import MeanReversionConfig, MeanReversionStrategy
from kryor.strategy.ml_signal import MLSignalConfig, MLSignalStrategy

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

    # ── Lookahead leak check for ML model ─────────────────
    from pathlib import Path as _Path
    model_path = _Path(__file__).parent.parent / "models" / "lgbm_signal_v1.pkl"
    if model_path.exists():
        try:
            from kryor.ml.trainer import load_model
            bundle = load_model(model_path)
            train_end = bundle.get("train_end")
            if train_end:
                train_end_dt = datetime.fromisoformat(train_end)
                if start < train_end_dt:
                    print("\n" + "!" * 70)
                    print(f"⚠️  LOOKAHEAD LEAK WARNING")
                    print(f"    Model trained until: {train_end}")
                    print(f"    Backtest starts at:  {start.date()}")
                    print(f"    Backtest period must START AFTER training END.")
                    print("!" * 70 + "\n")
                    if input("Continue anyway? (yes/no): ").lower() != "yes":
                        sys.exit(1)
                else:
                    print(f"\n✓ ML lookahead-safe: trained until {train_end}, backtest from {start.date()}")
        except Exception as e:
            print(f"Could not verify model train period: {e}")

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

    # ── Regime Actor ──────────────────────────────────────

    regime = RegimeBacktestActor(
        config=RegimeBacktestConfig(
            component_id="REGIME-BT",
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
            lookback_days=252,
            fit_samples=500,
        ),
    )
    engine.add_actor(regime)

    # ── Circuit Breaker Actor ─────────────────────────────

    cb = CircuitBreakerActor(
        config=CircuitBreakerConfig(
            component_id="CB-BT",
            daily_loss_limit_pct=0.03,
            weekly_dd_limit_pct=0.08,
            monthly_dd_limit_pct=0.15,
            max_daily_trades=20,
            max_consecutive_losses=5,
        ),
    )
    engine.add_actor(cb)

    # ── Strategies ────────────────────────────────────────

    momentum = MomentumStrategy(
        config=MomentumConfig(
            strategy_id="MOM-BT",
            symbols=args.symbols,
            lookback_days=252,
            skip_days=21,
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

    ml_signal = MLSignalStrategy(
        config=MLSignalConfig(
            strategy_id="ML-BT",
            symbols=args.symbols,
            model_path="models/lgbm_signal_v1.pkl",
            buy_threshold=0.55,  # 高信頼度のみ
            max_hold_days=10,
        ),
    )
    engine.add_strategy(ml_signal)

    # ── Run ───────────────────────────────────────────────

    print("\nRunning backtest...")
    engine.run()

    # ── Results ───────────────────────────────────────────

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    # Final account
    account = engine.trader.generate_account_report(VENUE)
    final_equity = args.capital
    if account is not None and not account.empty:
        final_equity = float(account.iloc[-1]["total"]) if "total" in account.columns else args.capital
        total_return = (final_equity - args.capital) / args.capital * 100
        print(f"\nStarting capital: ${args.capital:,.2f}")
        print(f"Final equity:     ${final_equity:,.2f}")
        print(f"Total return:     {total_return:+.2f}%")
        print(f"Period:           {args.years} years")
        print(f"Annualized:       {((final_equity / args.capital) ** (1 / args.years) - 1) * 100:+.2f}%")

    fills = engine.trader.generate_order_fills_report()
    if fills is not None and not fills.empty:
        print(f"\nTotal fills: {len(fills)}")

    positions = engine.trader.generate_positions_report()
    win_rate = 0.0
    closed_pnl = None
    if positions is not None and not positions.empty:
        closed = positions[positions["is_closed"] == True] if "is_closed" in positions.columns else positions
        if len(closed) > 0 and "realized_pnl" in closed.columns:
            closed_pnl = closed["realized_pnl"].apply(
                lambda x: float(str(x).split()[0]) if x else 0.0
            )
            wins = closed_pnl[closed_pnl > 0]
            losses = closed_pnl[closed_pnl < 0]
            win_rate = len(wins) / len(closed) * 100 if len(closed) else 0.0
            print(f"\nClosed positions: {len(closed)}")
            print(f"  Wins:    {len(wins)} (avg +${wins.mean():.2f})" if len(wins) else "  Wins:    0")
            print(f"  Losses:  {len(losses)} (avg -${abs(losses.mean()):.2f})" if len(losses) else "  Losses:  0")
            print(f"  Win rate: {win_rate:.1f}%")
            print(f"  Total P&L: ${closed_pnl.sum():+,.2f}")

    # ── Generate chart image ──────────────────────────────
    try:
        _plot_results(account, fills, closed_pnl, args.capital, final_equity, args.years, win_rate)
    except Exception as e:
        print(f"\nChart generation failed: {e}")

    engine.dispose()
    print("\nBacktest complete.")


def _plot_results(account, fills, closed_pnl, capital, final_equity, years, win_rate):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import pandas as pd
    from pathlib import Path

    out_dir = Path(__file__).parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"backtest_{timestamp}.png"

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"KRYOR Backtest — {years}yr | Return: {(final_equity/capital - 1)*100:+.2f}% "
        f"| Win Rate: {win_rate:.1f}%",
        fontsize=14, fontweight="bold",
    )

    # [1] Equity curve
    ax = axes[0, 0]
    if account is not None and not account.empty and "total" in account.columns:
        equity_series = account["total"].astype(float)
        ax.plot(equity_series.index, equity_series.values, color="#2d7d2d", linewidth=2)
        ax.axhline(y=capital, color="gray", linestyle="--", alpha=0.5, label=f"Initial ${capital:,.0f}")
        ax.fill_between(equity_series.index, capital, equity_series.values,
                         where=equity_series.values >= capital,
                         color="green", alpha=0.15)
        ax.fill_between(equity_series.index, capital, equity_series.values,
                         where=equity_series.values < capital,
                         color="red", alpha=0.15)
        ax.legend()
    ax.set_title("Equity Curve")
    ax.set_ylabel("USD")
    ax.grid(alpha=0.3)

    # [2] Drawdown
    ax = axes[0, 1]
    if account is not None and not account.empty and "total" in account.columns:
        equity = account["total"].astype(float)
        peak = equity.cummax()
        dd = (equity - peak) / peak * 100
        ax.fill_between(dd.index, dd.values, 0, color="red", alpha=0.4)
        ax.plot(dd.index, dd.values, color="darkred", linewidth=1)
        max_dd = dd.min()
        ax.axhline(y=max_dd, color="darkred", linestyle="--", alpha=0.7,
                   label=f"Max DD: {max_dd:.2f}%")
        ax.legend()
    ax.set_title("Drawdown")
    ax.set_ylabel("%")
    ax.grid(alpha=0.3)

    # [3] P&L distribution
    ax = axes[1, 0]
    if closed_pnl is not None and len(closed_pnl) > 0:
        wins = closed_pnl[closed_pnl > 0].values
        losses = closed_pnl[closed_pnl < 0].values
        bins = 30
        ax.hist(wins, bins=bins, color="green", alpha=0.6, label=f"Wins ({len(wins)})", edgecolor="darkgreen")
        ax.hist(losses, bins=bins, color="red", alpha=0.6, label=f"Losses ({len(losses)})", edgecolor="darkred")
        ax.axvline(x=0, color="black", linewidth=1)
        ax.legend()
    ax.set_title("P&L Distribution per Trade")
    ax.set_xlabel("USD")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.3)

    # [4] Cumulative realized P&L
    ax = axes[1, 1]
    if closed_pnl is not None and len(closed_pnl) > 0:
        cum_pnl = closed_pnl.cumsum().values
        ax.plot(range(len(cum_pnl)), cum_pnl,
                color="#1565c0", linewidth=2, label=f"Cumulative ${cum_pnl[-1]:+,.0f}")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.fill_between(range(len(cum_pnl)), 0, cum_pnl,
                         where=cum_pnl >= 0, color="green", alpha=0.15)
        ax.fill_between(range(len(cum_pnl)), 0, cum_pnl,
                         where=cum_pnl < 0, color="red", alpha=0.15)
        ax.legend()
    ax.set_title("Cumulative Realized P&L")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("USD")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nChart saved: {out_path}")


if __name__ == "__main__":
    main()
