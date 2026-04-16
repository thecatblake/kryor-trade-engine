"""LightGBM training pipeline with walk-forward cross validation.

Trains a 3-class classifier (SELL/HOLD/BUY) on technical features.
Saves model to disk for inference.
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit

from kryor.ml.features import FEATURE_COLS, compute_features, make_target


DEFAULT_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "n_estimators": 500,
    "learning_rate": 0.02,
    "max_depth": 6,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbose": -1,
}


def fetch_training_data(symbols: list[str], years: int = 5) -> pd.DataFrame:
    """Fetch OHLCV data for multiple symbols."""
    end = datetime.now()
    from datetime import timedelta
    start = end - timedelta(days=years * 365)

    frames = []
    for sym in symbols:
        try:
            df = yf.Ticker(sym).history(start=start, end=end, auto_adjust=True)
            if df.empty:
                continue
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            df["symbol"] = sym
            frames.append(df)
        except Exception as e:
            print(f"Failed to fetch {sym}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_dataset(
    raw: pd.DataFrame, horizon: int = 5, threshold: float = 0.01
) -> tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix and target labels from raw OHLCV."""
    all_X = []
    all_y = []

    for sym, group in raw.groupby("symbol"):
        group = group.sort_values("date").reset_index(drop=True)
        feats = compute_features(group)
        target = make_target(group, horizon=horizon, threshold=threshold)
        feats["target"] = target
        feats = feats.dropna(subset=FEATURE_COLS + ["target"])
        all_X.append(feats[FEATURE_COLS])
        all_y.append(feats["target"])

    if not all_X:
        return pd.DataFrame(), pd.Series()
    X = pd.concat(all_X, ignore_index=True)
    y = pd.concat(all_y, ignore_index=True).astype(int)
    return X, y


def walk_forward_train(
    X: pd.DataFrame, y: pd.Series, n_splits: int = 5, params: dict | None = None
) -> tuple[lgb.LGBMClassifier, dict]:
    """Walk-forward CV training. Returns final model trained on full data."""
    params = params or DEFAULT_PARAMS

    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        cv_scores.append(acc)
        print(f"  Fold {fold + 1}: accuracy = {acc:.4f}")

    print(f"\nMean CV accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    # Final model on full data
    print("\nTraining final model on full dataset...")
    final_model = lgb.LGBMClassifier(**params)
    final_model.fit(X, y)

    metrics = {
        "cv_accuracy_mean": float(np.mean(cv_scores)),
        "cv_accuracy_std": float(np.std(cv_scores)),
        "cv_scores": cv_scores,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "target_distribution": y.value_counts().to_dict(),
    }
    return final_model, metrics


def save_model(model: lgb.LGBMClassifier, metrics: dict, path: str | Path) -> None:
    """Save model + metadata to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": model,
        "feature_cols": FEATURE_COLS,
        "metrics": metrics,
        "trained_at": datetime.now().isoformat(),
        "version": "1.0",
    }
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved model to {path}")


def load_model(path: str | Path) -> dict:
    """Load model bundle from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)
