#!/usr/bin/env python3
"""Fetch and cache historical data for backtesting/analysis.

Usage:
    python scripts/fetch_historical.py
    python scripts/fetch_historical.py --symbols AAPL MSFT --years 3
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kryor.data.fetcher import fetch_bars_df, fetch_macro_data
from kryor.data.indicators import compute_features

DEFAULT_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "JPM", "V", "JNJ", "UNH", "HD", "PG", "MA", "XOM",
    "SPY", "QQQ", "IWM",  # ETFs for benchmarking
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch historical market data")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--output", type=str, default="data")
    args = parser.parse_args()

    outdir = Path(args.output)
    outdir.mkdir(exist_ok=True)

    print(f"Fetching {len(args.symbols)} symbols, {args.years} years of data...")

    # Fetch OHLCV
    bars = fetch_bars_df(args.symbols, years=args.years)
    if bars.empty:
        print("ERROR: No data fetched")
        sys.exit(1)

    # Add indicators
    frames = []
    for sym in args.symbols:
        sym_df = bars[bars["symbol"] == sym].copy()
        if len(sym_df) > 50:
            sym_df = compute_features(sym_df)
            frames.append(sym_df)

    if frames:
        import pandas as pd
        all_data = pd.concat(frames)
        path = outdir / "historical_bars.parquet"
        all_data.to_parquet(path, index=False)
        print(f"Saved {len(all_data)} bars to {path}")

    # Fetch macro data
    macro = fetch_macro_data(years=args.years)
    if not macro.empty:
        path = outdir / "macro_data.parquet"
        macro.to_parquet(path)
        print(f"Saved {len(macro)} macro rows to {path}")

    print("Done!")


if __name__ == "__main__":
    main()
