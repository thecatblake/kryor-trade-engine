"""Feature engineering for ML signals.

14 technical features per design spec, computed from OHLCV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLS = [
    "rsi_14", "rsi_28",
    "bb_width", "bb_pctb",
    "atr_14_norm",
    "ema_cross_5_20", "sma_cross_10_50",
    "volume_ratio",
    "roc_10",
    "vol_20", "vol_60",
    "momentum_60", "momentum_120",
    "high_low_range_5",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 14 features from OHLCV DataFrame.

    Expects columns: open, high, low, close, volume (lowercase).
    Returns DataFrame with feature columns added.
    """
    out = df.copy()
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    volume = out["volume"].astype(float)

    # RSI
    out["rsi_14"] = _rsi(close, 14)
    out["rsi_28"] = _rsi(close, 28)

    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    out["bb_width"] = (upper - lower) / sma20
    out["bb_pctb"] = (close - lower) / (upper - lower).replace(0, np.nan)

    # ATR (normalized by price)
    atr = _atr(high, low, close, 14)
    out["atr_14_norm"] = atr / close

    # EMA / SMA crosses (normalized differences)
    ema5 = close.ewm(span=5, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    out["ema_cross_5_20"] = (ema5 - ema20) / ema20

    sma10 = close.rolling(10).mean()
    sma50 = close.rolling(50).mean()
    out["sma_cross_10_50"] = (sma10 - sma50) / sma50

    # Volume ratio
    vol_ma20 = volume.rolling(20).mean()
    out["volume_ratio"] = volume / vol_ma20.replace(0, np.nan)

    # Rate of Change
    out["roc_10"] = close.pct_change(10)

    # Volatility (rolling std of returns)
    returns = close.pct_change()
    out["vol_20"] = returns.rolling(20).std()
    out["vol_60"] = returns.rolling(60).std()

    # Long-term momentum
    out["momentum_60"] = close.pct_change(60)
    out["momentum_120"] = close.pct_change(120)

    # High-low range over 5 days
    out["high_low_range_5"] = (high.rolling(5).max() - low.rolling(5).min()) / close

    return out


def make_target(df: pd.DataFrame, horizon: int = 5, threshold: float = 0.01) -> pd.Series:
    """Create classification target.

    Returns:
        2 if future_return > +threshold (BUY)
        0 if future_return < -threshold (SELL)
        1 otherwise (HOLD)
    """
    future_return = df["close"].shift(-horizon) / df["close"] - 1
    target = pd.Series(1, index=df.index)  # HOLD by default
    target[future_return > threshold] = 2  # BUY
    target[future_return < -threshold] = 0  # SELL
    return target


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()
