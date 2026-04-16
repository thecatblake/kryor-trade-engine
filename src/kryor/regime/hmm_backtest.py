"""Backtest-compatible RegimeActor.

Pre-fetches all macro data, fits HMM on the first N days,
then publishes regime updates aligned to bar timestamps.
No network calls during the backtest loop.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from hmmlearn.hmm import GaussianHMM

from nautilus_trader.common.actor import Actor
from nautilus_trader.config import ActorConfig
from nautilus_trader.model.data import Bar, BarSpecification, BarType
from nautilus_trader.model.enums import AggregationSource, BarAggregation, PriceType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue

from kryor.core.custom_data import RegimeData, RegimeEnum

REGIME_TOPIC = "regime.update"
SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLC", "XLY", "XLP", "XLU", "XLRE", "XLB"]


class RegimeBacktestConfig(ActorConfig, frozen=True):
    start_date: str  # e.g. "2022-01-01"
    end_date: str    # e.g. "2026-04-15"
    trigger_symbol: str = "SPY"  # このシンボルのバー受信でregime判定
    lookback_days: int = 252
    fit_samples: int = 500  # Use first N days for HMM fit
    min_confidence: float = 0.7  # 確信度がこれ未満ならregime変更しない
    min_persistence_days: int = 5  # 同じregimeが最低N日続いてから確定


class RegimeBacktestActor(Actor):
    """Regime detection that works in BacktestEngine (no live data fetching)."""

    def __init__(self, config: RegimeBacktestConfig) -> None:
        super().__init__(config=config)
        self._config = config
        self._hmm = GaussianHMM(
            n_components=3, covariance_type="full",
            n_iter=100, random_state=42, init_params="stmc",
        )
        self._state_map: dict[int, RegimeEnum] = {}
        self._fitted = False
        self._last_regime: RegimeEnum = RegimeEnum.NEUTRAL
        self._published_regime: RegimeEnum = RegimeEnum.NEUTRAL
        self._candidate_regime: RegimeEnum | None = None
        self._candidate_streak: int = 0
        self._macro_df: pd.DataFrame = pd.DataFrame()
        self._features_by_date: dict[pd.Timestamp, np.ndarray] = {}
        self._last_published_date: pd.Timestamp | None = None

    def on_start(self) -> None:
        self.log.info("RegimeBacktestActor starting — fetching historical macro data")

        start = datetime.fromisoformat(self._config.start_date)
        end = datetime.fromisoformat(self._config.end_date)

        # Fetch all data with extra lookback for z-score normalization
        fetch_start = start - timedelta(days=self._config.lookback_days + 60)
        self._macro_df = self._fetch_macro(fetch_start, end)

        if self._macro_df.empty:
            self.log.warning("No macro data, regime will stay NEUTRAL")
            return

        # Fit HMM on first fit_samples days
        features_full = self._prepare_features(self._macro_df)
        if len(features_full) < 60:
            self.log.warning(f"Not enough samples ({len(features_full)}) for HMM")
            return

        n_fit = min(self._config.fit_samples, len(features_full) - 50)
        fit_features = features_full[:n_fit]
        self._hmm.fit(fit_features)
        self._map_states_to_regimes(fit_features)
        self._fitted = True
        self.log.info(f"HMM fitted on {n_fit} samples. Map: {self._state_map}")

        # Pre-compute features for each date
        valid_dates = self._macro_df.dropna().index
        prepared = self._prepare_features_with_dates(self._macro_df)
        for date, feat in prepared.items():
            self._features_by_date[date] = feat

        # Subscribe to trigger symbol bars so on_bar fires
        instrument_id = InstrumentId(Symbol(self._config.trigger_symbol), Venue("ALPACA"))
        bar_spec = BarSpecification(1, BarAggregation.DAY, PriceType.LAST)
        bar_type = BarType(instrument_id, bar_spec, AggregationSource.EXTERNAL)
        self.subscribe_bars(bar_type)
        self.log.info(f"Subscribed to {self._config.trigger_symbol} bars for regime updates")

    def on_bar(self, bar: Bar) -> None:
        """On each bar, publish the regime for that date."""
        if not self._fitted:
            return

        ts_ns = bar.ts_event
        bar_date = pd.Timestamp(ts_ns, unit="ns").normalize()

        # Only publish once per day
        if self._last_published_date == bar_date:
            return

        # Find closest feature row at or before bar_date
        available_dates = [d for d in self._features_by_date.keys() if d <= bar_date]
        if not available_dates:
            return
        feat_date = max(available_dates)
        features = self._features_by_date[feat_date]

        try:
            # Run inference on the features up to this date
            # We need all features up to feat_date for state sequence
            all_dates = sorted([d for d in self._features_by_date.keys() if d <= feat_date])
            feat_matrix = np.array([self._features_by_date[d] for d in all_dates])
            proba = self._hmm.predict_proba(feat_matrix)
            state_seq = self._hmm.predict(feat_matrix)

            current_state = int(state_seq[-1])
            current_proba = proba[-1]
            raw_regime = self._state_map.get(current_state, RegimeEnum.NEUTRAL)
            confidence = float(current_proba[current_state])

            # 確信度フィルター+持続性フィルターで published regime を決定
            if raw_regime != self._published_regime:
                if confidence < self._config.min_confidence:
                    raw_regime = self._published_regime  # 確信低い → 据え置き
                elif self._candidate_regime == raw_regime:
                    self._candidate_streak += 1
                    if self._candidate_streak < self._config.min_persistence_days:
                        raw_regime = self._published_regime  # 持続不足 → 据え置き
                else:
                    self._candidate_regime = raw_regime
                    self._candidate_streak = 1
                    raw_regime = self._published_regime  # 初検出 → 据え置き

            if raw_regime != self._published_regime:
                self.log.info(
                    f"{bar_date.date()}: Regime changed to {raw_regime.name} "
                    f"(p={confidence:.2f}, streak={self._candidate_streak})"
                )
                self._published_regime = raw_regime
                self._candidate_regime = None
                self._candidate_streak = 0

            bull_idx = self._regime_to_state(RegimeEnum.BULL)
            neutral_idx = self._regime_to_state(RegimeEnum.NEUTRAL)
            bear_idx = self._regime_to_state(RegimeEnum.BEAR)

            data = RegimeData(
                regime=self._published_regime,
                probability=confidence,
                bull_prob=float(current_proba[bull_idx]) if bull_idx is not None else 0.0,
                neutral_prob=float(current_proba[neutral_idx]) if neutral_idx is not None else 0.0,
                bear_prob=float(current_proba[bear_idx]) if bear_idx is not None else 0.0,
                ts_event=ts_ns,
                ts_init=ts_ns,
            )
            self.msgbus.publish(REGIME_TOPIC, data)
            self._last_published_date = bar_date
        except Exception as e:
            self.log.error(f"Inference failed: {e}")

    # ── Data Prep (same logic as live hmm.py) ──────────────

    def _fetch_macro(self, start: datetime, end: datetime) -> pd.DataFrame:
        frames = {}
        try:
            vix = yf.Ticker("^VIX").history(start=start, end=end, auto_adjust=True)["Close"]
            vix.index = vix.index.tz_localize(None).normalize()
            frames["vix"] = vix
        except Exception as e:
            self.log.warning(f"VIX fetch failed: {e}")
        try:
            spy = yf.Ticker("SPY").history(start=start, end=end, auto_adjust=True)
            spy.index = spy.index.tz_localize(None).normalize()
            vol_20 = spy["Volume"].rolling(20).mean()
            vol_60 = spy["Volume"].rolling(60).mean()
            frames["volume_change"] = (vol_20 / vol_60) - 1
        except Exception as e:
            self.log.warning(f"SPY fetch failed: {e}")
        try:
            sector = yf.download(SECTOR_ETFS, start=start, end=end, auto_adjust=True)["Close"]
            sector.index = sector.index.tz_localize(None).normalize()
            frames["sector_dispersion"] = sector.pct_change(20).std(axis=1)
        except Exception as e:
            self.log.warning(f"Sector fetch failed: {e}")
        try:
            tnx = yf.Ticker("^TNX").history(start=start, end=end, auto_adjust=True)["Close"]
            irx = yf.Ticker("^IRX").history(start=start, end=end, auto_adjust=True)["Close"]
            tnx.index = tnx.index.tz_localize(None).normalize()
            irx.index = irx.index.tz_localize(None).normalize()
            frames["yield_curve_slope"] = tnx - irx
        except Exception as e:
            self.log.warning(f"Yield curve fetch failed: {e}")

        macro = pd.DataFrame(frames).ffill().dropna(how="all")
        if "vix" in macro.columns:
            macro["credit_spread"] = macro["vix"] / 10
        return macro

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        cols = ["vix", "volume_change", "sector_dispersion", "yield_curve_slope", "credit_spread"]
        feat = pd.DataFrame(index=df.index)
        for col in cols:
            s = df[col].astype(float) if col in df.columns else pd.Series(0.0, index=df.index)
            rm = s.rolling(self._config.lookback_days, min_periods=20).mean()
            rs = s.rolling(self._config.lookback_days, min_periods=20).std().replace(0, 1)
            feat[col] = (s - rm) / rs
        return feat.dropna().values

    def _prepare_features_with_dates(self, df: pd.DataFrame) -> dict[pd.Timestamp, np.ndarray]:
        cols = ["vix", "volume_change", "sector_dispersion", "yield_curve_slope", "credit_spread"]
        feat = pd.DataFrame(index=df.index)
        for col in cols:
            s = df[col].astype(float) if col in df.columns else pd.Series(0.0, index=df.index)
            rm = s.rolling(self._config.lookback_days, min_periods=20).mean()
            rs = s.rolling(self._config.lookback_days, min_periods=20).std().replace(0, 1)
            feat[col] = (s - rm) / rs
        feat = feat.dropna()
        return {date: row.values for date, row in feat.iterrows()}

    def _map_states_to_regimes(self, features: np.ndarray) -> None:
        states = self._hmm.predict(features)
        state_vix_means = {s: features[states == s, 0].mean() if (states == s).any() else 0.0
                          for s in range(3)}
        sorted_states = sorted(state_vix_means.keys(), key=lambda s: state_vix_means[s])
        self._state_map = {
            sorted_states[0]: RegimeEnum.BULL,
            sorted_states[1]: RegimeEnum.NEUTRAL,
            sorted_states[2]: RegimeEnum.BEAR,
        }

    def _regime_to_state(self, regime: RegimeEnum) -> int | None:
        for state, r in self._state_map.items():
            if r == regime:
                return state
        return None
