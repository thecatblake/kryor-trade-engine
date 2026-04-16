"""Technical indicator computation. Uses pandas-ta or manual calc."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 14 technical features from OHLCV data.

    Expects columns: open, high, low, close, volume (lowercase).
    Returns DataFrame with feature columns added.
    """
    out = df.copy()

    # RSI
    out["rsi_14"] = _rsi(out["close"], 14)
    out["rsi_28"] = _rsi(out["close"], 28)

    # Bollinger Bands
    sma20 = out["close"].rolling(20).mean()
    std20 = out["close"].rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    out["bb_width"] = (upper - lower) / sma20
    out["bb_pctb"] = (out["close"] - lower) / (upper - lower)

    # ATR
    out["atr_14"] = _atr(out["high"], out["low"], out["close"], 14)

    # EMA cross
    ema5 = out["close"].ewm(span=5, adjust=False).mean()
    ema20 = out["close"].ewm(span=20, adjust=False).mean()
    out["ema_cross_5_20"] = (ema5 - ema20) / ema20

    # SMA cross
    sma10 = out["close"].rolling(10).mean()
    sma50 = out["close"].rolling(50).mean()
    out["sma_cross_10_50"] = (sma10 - sma50) / sma50

    # Volume ratio
    vol_ma20 = out["volume"].rolling(20).mean()
    out["volume_ratio"] = out["volume"] / vol_ma20.replace(0, np.nan)

    # Rate of Change
    out["roc_10"] = out["close"].pct_change(10)

    return out


def compute_momentum_score(df: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.Series:
    """12-1 month momentum score, volatility-adjusted."""
    total_return = df["close"].pct_change(lookback)
    recent_return = df["close"].pct_change(skip)
    raw_momentum = total_return - recent_return
    atr = _atr(df["high"], df["low"], df["close"], 20)
    return raw_momentum / (atr / df["close"]).replace(0, np.nan)


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()
