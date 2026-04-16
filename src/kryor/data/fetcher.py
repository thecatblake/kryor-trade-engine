"""Historical data fetching via yfinance. Swap this module to change data source."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

from kryor.core.models import NormalizedBar


def fetch_bars(
    symbol: str,
    start: datetime | None = None,
    end: datetime | None = None,
    timeframe: str = "1d",
    years: int = 5,
) -> list[NormalizedBar]:
    if start is None:
        start = datetime.now() - timedelta(days=years * 365)
    if end is None:
        end = datetime.now()

    logger.info(f"Fetching {symbol} {timeframe} bars from {start:%Y-%m-%d} to {end:%Y-%m-%d}")
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, interval=timeframe, auto_adjust=True)

    if df.empty:
        logger.warning(f"No data for {symbol}")
        return []

    bars = []
    for ts, row in df.iterrows():
        bars.append(
            NormalizedBar(
                timestamp=ts.to_pydatetime(),
                symbol=symbol,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=int(row["Volume"]),
                vwap=float((row["High"] + row["Low"] + row["Close"]) / 3),
                timeframe=timeframe,
            )
        )
    logger.info(f"Fetched {len(bars)} bars for {symbol}")
    return bars


def fetch_bars_df(
    symbols: list[str],
    start: datetime | None = None,
    end: datetime | None = None,
    years: int = 5,
) -> pd.DataFrame:
    """Fetch OHLCV for multiple symbols, return a DataFrame with MultiIndex (date, symbol)."""
    if start is None:
        start = datetime.now() - timedelta(days=years * 365)
    if end is None:
        end = datetime.now()

    logger.info(f"Fetching {len(symbols)} symbols: {symbols[:5]}...")
    data = yf.download(symbols, start=start, end=end, auto_adjust=True, group_by="ticker")

    frames = []
    for sym in symbols:
        try:
            if len(symbols) == 1:
                df = data.copy()
            else:
                df = data[sym].copy()
            df = df.dropna(subset=["Close"])
            df["symbol"] = sym
            frames.append(df)
        except (KeyError, TypeError):
            logger.warning(f"No data for {sym}, skipping")
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames).reset_index()
    result.columns = [c.lower() if isinstance(c, str) else c for c in result.columns]
    return result


def fetch_macro_data(start: datetime | None = None, years: int = 5) -> pd.DataFrame:
    """Fetch VIX, yield curve (10Y-2Y), credit spread proxies via yfinance."""
    if start is None:
        start = datetime.now() - timedelta(days=years * 365)

    tickers = {
        "^VIX": "vix",
        "^TNX": "us10y",  # 10-year treasury yield
        "^IRX": "us3m",   # 3-month treasury
    }
    frames = {}
    for ticker, name in tickers.items():
        try:
            df = yf.Ticker(ticker).history(start=start, auto_adjust=True)
            frames[name] = df["Close"]
        except Exception as e:
            logger.warning(f"Failed to fetch {ticker}: {e}")

    macro = pd.DataFrame(frames)
    macro.index = pd.to_datetime(macro.index)

    if "us10y" in macro.columns and "us3m" in macro.columns:
        macro["yield_curve_slope"] = macro["us10y"] - macro["us3m"]

    # Sector ETF dispersion
    sector_etfs = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLC", "XLY", "XLP", "XLU", "XLRE", "XLB"]
    try:
        sector_data = yf.download(sector_etfs, start=start, auto_adjust=True)["Close"]
        returns_20d = sector_data.pct_change(20)
        macro["sector_dispersion"] = returns_20d.std(axis=1)
    except Exception as e:
        logger.warning(f"Failed to fetch sector ETFs: {e}")

    return macro.ffill().dropna(how="all")
