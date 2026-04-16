#!/usr/bin/env python3
"""Train ML signal model and save to models/ directory.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --symbols AAPL MSFT --years 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kryor.ml.trainer import (
    build_dataset,
    fetch_training_data,
    save_model,
    walk_forward_train,
)

DEFAULT_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "JPM", "V", "JNJ", "UNH", "HD", "PG", "MA", "XOM", "LLY", "ABBV",
    "MRK", "COST", "AVGO", "PEP", "KO", "TMO", "WMT", "ADBE", "CRM",
    "ACN", "MCD", "NKE", "LIN", "TXN", "AMD", "QCOM", "ISRG", "AMAT",
    "SPY", "QQQ", "IWM",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM signal model")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=5,
                        help="Days ahead for return target")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Return threshold for BUY/SELL classification")
    parser.add_argument("--output", type=str, default="models/lgbm_signal_v1.pkl")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    output_path = project_root / args.output

    print(f"Fetching {args.years} years of data for {len(args.symbols)} symbols...")
    raw = fetch_training_data(args.symbols, years=args.years)
    print(f"Total raw rows: {len(raw)}")

    print(f"\nBuilding features (horizon={args.horizon}d, threshold={args.threshold})...")
    X, y = build_dataset(raw, horizon=args.horizon, threshold=args.threshold)
    print(f"Feature matrix: {X.shape}")
    print(f"Target distribution:\n{y.value_counts().sort_index()}")

    print("\nTraining with walk-forward CV...")
    model, metrics = walk_forward_train(X, y, n_splits=5)

    print("\nFeature importance (top 10):")
    importance = model.feature_importances_
    for name, imp in sorted(
        zip(X.columns, importance), key=lambda x: x[1], reverse=True
    )[:10]:
        print(f"  {name:25s} {imp}")

    save_model(model, metrics, output_path)
    print(f"\nDone. Model saved: {output_path}")


if __name__ == "__main__":
    main()
