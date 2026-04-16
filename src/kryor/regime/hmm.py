"""HMM 3-state regime detection — NautilusTrader Actor.

On startup:
  1. Fetches 5 years of macro data (VIX, yield curve, sector dispersion) via yfinance
  2. Computes SPY volume change from historical data
  3. Fits GaussianHMM on all 5 features
  4. Publishes initial regime

On daily timer:
  1. Re-fetches latest macro data
  2. Runs HMM inference with all 5 features
  3. Publishes updated RegimeData to strategies via MessageBus
"""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM

from nautilus_trader.common.actor import Actor
from nautilus_trader.config import ActorConfig

from kryor.core.custom_data import RegimeData, RegimeEnum

REGIME_TOPIC = "regime.update"

SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLC", "XLY", "XLP", "XLU", "XLRE", "XLB"]


class RegimeActorConfig(ActorConfig, frozen=True):
    n_states: int = 3
    lookback_days: int = 252
    min_samples: int = 60
    history_years: int = 5
    update_interval_hours: float = 6.0  # Re-evaluate every N hours


class RegimeActor(Actor):
    """NT Actor — full 5-feature HMM regime detection.

    Features:
      1. VIX z-score
      2. SPY volume change (20d/60d ratio - 1)
      3. Sector return dispersion (std of 11 sector ETF 20d returns)
      4. Yield curve slope (10Y - 3M treasury)
      5. Credit spread proxy (BAA-equivalent, approximated from VIX)
    """

    def __init__(self, config: RegimeActorConfig) -> None:
        super().__init__(config=config)
        self._config = config
        self._hmm = GaussianHMM(
            n_components=config.n_states,
            covariance_type="full",
            n_iter=100,
            random_state=42,
            init_params="stmc",
        )
        self._state_map: dict[int, RegimeEnum] = {}
        self._fitted = False
        self._regime_start_ts: int = 0
        self._last_regime: RegimeEnum = RegimeEnum.NEUTRAL
        self._macro_df: pd.DataFrame = pd.DataFrame()

    def on_start(self) -> None:
        self.log.info("RegimeActor starting — fetching macro data and fitting HMM")

        # Fetch and fit
        try:
            self._macro_df = self._fetch_all_macro_data(years=self._config.history_years)
            if not self._macro_df.empty:
                self._fit_hmm(self._macro_df)
                self._run_inference()
            else:
                self.log.warning("No macro data fetched, starting with NEUTRAL regime")
                self._publish_default()
        except Exception as e:
            self.log.error(f"Regime init failed: {e}")
            self._publish_default()

        # Schedule periodic re-evaluation
        from datetime import timedelta as td
        self.clock.set_timer(
            name="regime_update",
            interval=td(hours=self._config.update_interval_hours),
            callback=self._on_timer,
        )
        self.log.info(
            f"RegimeActor ready — fitted={self._fitted}, "
            f"update every {self._config.update_interval_hours}h"
        )

    def _on_timer(self, event) -> None:
        """Periodic macro data refresh and regime re-evaluation."""
        try:
            fresh = self._fetch_all_macro_data(years=2)
            if not fresh.empty:
                self._macro_df = fresh
                if not self._fitted:
                    self._fit_hmm(self._macro_df)
                self._run_inference()
        except Exception as e:
            self.log.error(f"Regime update failed: {e}")

    # ── Data Fetching ──────────────────────────────────────

    def _fetch_all_macro_data(self, years: int = 5) -> pd.DataFrame:
        """Fetch all 5 features from yfinance."""
        start = datetime.now() - timedelta(days=years * 365)
        self.log.info(f"Fetching macro data from {start:%Y-%m-%d}...")

        frames = {}

        # [1] VIX
        try:
            vix = yf.Ticker("^VIX").history(start=start, auto_adjust=True)["Close"]
            vix.index = vix.index.tz_localize(None).normalize()
            frames["vix"] = vix
            self.log.info(f"  VIX: {len(vix)} rows")
        except Exception as e:
            self.log.warning(f"  VIX fetch failed: {e}")

        # [2] SPY volume change (20d / 60d - 1)
        try:
            spy = yf.Ticker("SPY").history(start=start, auto_adjust=True)
            spy.index = spy.index.tz_localize(None).normalize()
            vol_20 = spy["Volume"].rolling(20).mean()
            vol_60 = spy["Volume"].rolling(60).mean()
            frames["volume_change"] = (vol_20 / vol_60) - 1
            self.log.info(f"  SPY volume: {len(spy)} rows")
        except Exception as e:
            self.log.warning(f"  SPY volume fetch failed: {e}")

        # [3] Sector return dispersion
        try:
            sector_data = yf.download(SECTOR_ETFS, start=start, auto_adjust=True)["Close"]
            sector_data.index = sector_data.index.tz_localize(None).normalize()
            returns_20d = sector_data.pct_change(20)
            frames["sector_dispersion"] = returns_20d.std(axis=1)
            self.log.info(f"  Sector dispersion: {len(sector_data)} rows")
        except Exception as e:
            self.log.warning(f"  Sector dispersion fetch failed: {e}")

        # [4] Yield curve slope (10Y - 3M)
        try:
            tnx = yf.Ticker("^TNX").history(start=start, auto_adjust=True)["Close"]
            irx = yf.Ticker("^IRX").history(start=start, auto_adjust=True)["Close"]
            tnx.index = tnx.index.tz_localize(None).normalize()
            irx.index = irx.index.tz_localize(None).normalize()
            frames["yield_curve_slope"] = tnx - irx
            self.log.info(f"  Yield curve: {len(tnx)} rows")
        except Exception as e:
            self.log.warning(f"  Yield curve fetch failed: {e}")

        # Combine all into single DataFrame with aligned dates
        macro = pd.DataFrame(frames)
        macro = macro.ffill().dropna(how="all")

        # [5] Credit spread proxy
        if "vix" in macro.columns:
            macro["credit_spread"] = macro["vix"] / 10

        self.log.info(f"Macro data: {len(macro)} rows, columns={list(macro.columns)}")
        return macro

    # ── HMM Fit & Inference ────────────────────────────────

    def _fit_hmm(self, df: pd.DataFrame) -> None:
        features = self._prepare_features(df)
        if len(features) < self._config.min_samples:
            self.log.warning(f"Only {len(features)} samples, need >= {self._config.min_samples}")
            return

        self._hmm.fit(features)
        self._map_states_to_regimes(features)
        self._fitted = True
        self.log.info(f"HMM fitted on {len(features)} samples. Map: {self._state_map}")

    def _run_inference(self) -> None:
        if not self._fitted or self._macro_df.empty:
            self._publish_default()
            return

        features = self._prepare_features(self._macro_df)
        if len(features) == 0:
            self._publish_default()
            return

        try:
            proba = self._hmm.predict_proba(features)
            state_seq = self._hmm.predict(features)
            current_state = int(state_seq[-1])
            current_proba = proba[-1]

            regime = self._state_map.get(current_state, RegimeEnum.NEUTRAL)

            ts = int(time.time() * 1e9)
            if regime != self._last_regime:
                self._regime_start_ts = ts
                self._last_regime = regime
            duration = int((ts - self._regime_start_ts) / (86400 * 1e9)) if self._regime_start_ts else 0

            bull_idx = self._regime_to_state(RegimeEnum.BULL)
            neutral_idx = self._regime_to_state(RegimeEnum.NEUTRAL)
            bear_idx = self._regime_to_state(RegimeEnum.BEAR)

            data = RegimeData(
                regime=regime,
                probability=float(current_proba[current_state]),
                bull_prob=float(current_proba[bull_idx]) if bull_idx is not None else 0.0,
                neutral_prob=float(current_proba[neutral_idx]) if neutral_idx is not None else 0.0,
                bear_prob=float(current_proba[bear_idx]) if bear_idx is not None else 0.0,
                duration_days=duration,
                ts_event=ts,
                ts_init=ts,
            )
            self.msgbus.publish(REGIME_TOPIC, data)
            self.log.info(
                f"Regime: {regime.name} (p={data.probability:.2f}, "
                f"bull={data.bull_prob:.2f}, neutral={data.neutral_prob:.2f}, "
                f"bear={data.bear_prob:.2f}, duration={duration}d)"
            )
        except Exception as e:
            self.log.error(f"HMM inference failed: {e}")
            self._publish_default()

    # ── Feature Preparation ────────────────────────────────

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Normalize all 5 features to z-scores using 252-day rolling window."""
        cols = ["vix", "volume_change", "sector_dispersion", "yield_curve_slope", "credit_spread"]
        feature_df = pd.DataFrame(index=df.index)
        lookback = self._config.lookback_days

        for col in cols:
            if col in df.columns:
                series = df[col].astype(float)
            else:
                series = pd.Series(0.0, index=df.index)

            rolling_mean = series.rolling(lookback, min_periods=20).mean()
            rolling_std = series.rolling(lookback, min_periods=20).std().replace(0, 1)
            feature_df[col] = (series - rolling_mean) / rolling_std

        feature_df = feature_df.dropna()
        return feature_df.values

    def _map_states_to_regimes(self, features: np.ndarray) -> None:
        states = self._hmm.predict(features)
        vix_idx = 0
        state_vix_means = {}
        for s in range(self._config.n_states):
            mask = states == s
            if mask.any():
                state_vix_means[s] = features[mask, vix_idx].mean()
            else:
                state_vix_means[s] = 0.0

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

    def _publish_default(self) -> None:
        ts = int(time.time() * 1e9)
        data = RegimeData(
            regime=RegimeEnum.NEUTRAL,
            probability=0.5,
            bull_prob=0.25,
            neutral_prob=0.50,
            bear_prob=0.25,
            ts_event=ts,
            ts_init=ts,
        )
        self.msgbus.publish(REGIME_TOPIC, data)
