"""ML-based signal strategy — uses trained LightGBM model.

Loads model on start, predicts signals from technical features each bar.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarSpecification, BarType
from nautilus_trader.model.enums import (
    AggregationSource,
    BarAggregation,
    OrderSide,
    PriceType,
    TimeInForce,
)
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Currency, Price, Quantity
from nautilus_trader.trading.strategy import Strategy

from kryor.adapters.alpaca.constants import ALPACA_VENUE
from kryor.core.custom_data import RegimeData, RegimeEnum
from kryor.ml.features import FEATURE_COLS, compute_features
from kryor.ml.predictor import MLPredictor

_USD_CURRENCY = Currency.from_str("USD")


class MLSignalConfig(StrategyConfig, frozen=True):
    symbols: list[str] = []
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    model_path: str = "models/lgbm_signal_v1.pkl"
    buy_threshold: float = 0.45
    sell_threshold: float = 0.45
    min_bars_required: int = 130  # 120日momentumに必要
    risk_per_trade_pct: float = 0.01
    max_position_pct: float = 0.10
    max_hold_days: int = 10
    stop_atr_mult: float = 2.0


class MLSignalStrategy(Strategy):
    """LightGBM-based signal generator strategy."""

    def __init__(self, config: MLSignalConfig) -> None:
        super().__init__(config=config)
        self._config = config
        self._regime = RegimeEnum.NEUTRAL
        self._regime_kelly = 0.25
        self._cb_level = 0
        self._bars: dict[str, deque[Bar]] = {}
        self._entry_bars: dict[str, int] = {}
        self._stop_loss: dict[str, float] = {}
        self._total_bars = 0
        self._predictor: MLPredictor | None = None

    def _on_regime_update(self, data: RegimeData) -> None:
        prev = self._regime
        self._regime = data.regime
        self._regime_kelly = data.kelly_fraction
        if data.regime == RegimeEnum.BEAR and prev != RegimeEnum.BEAR:
            for position in self.cache.positions_open(strategy_id=self.id):
                self.close_position(position)
            self._stop_loss.clear()
            self._entry_bars.clear()

    def _on_circuit_breaker(self, data) -> None:
        self._cb_level = data.level
        if data.level >= 3:
            for position in self.cache.positions_open(strategy_id=self.id):
                self.close_position(position)
            self._stop_loss.clear()
            self._entry_bars.clear()

    def on_start(self) -> None:
        self.log.info("MLSignalStrategy starting")
        self.msgbus.subscribe("regime.update", self._on_regime_update)
        self.msgbus.subscribe("circuit_breaker.update", self._on_circuit_breaker)

        # Load model
        model_path = Path(self._config.model_path)
        if not model_path.is_absolute():
            # Try relative to project root
            for candidate in [Path.cwd() / model_path,
                              Path(__file__).parent.parent.parent.parent / model_path]:
                if candidate.exists():
                    model_path = candidate
                    break

        if not model_path.exists():
            self.log.error(f"Model not found: {model_path}")
            return

        try:
            self._predictor = MLPredictor(model_path)
            self.log.info(
                f"Loaded ML model: {model_path.name} "
                f"(trained {self._predictor.trained_at[:10]}, "
                f"CV acc={self._predictor.metrics.get('cv_accuracy_mean', 0):.3f})"
            )
        except Exception as e:
            self.log.error(f"Failed to load model: {e}")
            return

        # Preload historical bars
        if self._config.alpaca_api_key:
            from kryor.adapters.alpaca.history import fetch_historical_bars
            for sym in self._config.symbols:
                self._bars[sym] = deque(maxlen=300)
                hist = fetch_historical_bars(
                    self._config.alpaca_api_key,
                    self._config.alpaca_secret_key,
                    sym, days=300,
                )
                for bar in hist:
                    self._bars[sym].append(bar)
            self.log.info(f"Preloaded ~{len(self._bars.get(self._config.symbols[0], []))} bars/symbol")
        else:
            for sym in self._config.symbols:
                self._bars[sym] = deque(maxlen=300)

        for sym in self._config.symbols:
            instrument_id = InstrumentId.from_str(f"{sym}.ALPACA")
            bar_spec = BarSpecification(1, BarAggregation.DAY, PriceType.LAST)
            bar_type = BarType(instrument_id, bar_spec, AggregationSource.EXTERNAL)
            self.subscribe_bars(bar_type)

    def on_bar(self, bar: Bar) -> None:
        sym = bar.bar_type.instrument_id.symbol.value
        if sym not in self._bars:
            return
        self._bars[sym].append(bar)
        self._total_bars += 1

        # Stop loss (every bar)
        self._check_stop_loss(sym, float(bar.close))

        if self._regime == RegimeEnum.BEAR:
            return
        if self._cb_level >= 2:
            return
        if self._predictor is None:
            return

        # Time-based exit
        self._check_time_exit(sym)

        # ML signal
        self._check_ml_signal(sym)

    def _check_ml_signal(self, sym: str) -> None:
        bars = self._bars[sym]
        if len(bars) < self._config.min_bars_required:
            return

        instrument_id = InstrumentId.from_str(f"{sym}.ALPACA")
        if self.portfolio.is_net_long(instrument_id):
            return

        # Build feature row
        df = pd.DataFrame([{
            "open": float(b.open),
            "high": float(b.high),
            "low": float(b.low),
            "close": float(b.close),
            "volume": float(b.volume),
        } for b in bars])

        feats = compute_features(df).dropna(subset=FEATURE_COLS)
        if feats.empty:
            return

        last_row = feats.iloc[-1]
        signal, confidence = self._predictor.predict_signal(
            last_row,
            buy_threshold=self._config.buy_threshold,
            sell_threshold=self._config.sell_threshold,
        )

        if signal != "buy":
            return

        # Position sizing
        account = self.portfolio.account(ALPACA_VENUE)
        if account is None:
            return
        equity = float(account.balance_total(account.base_currency or _USD_CURRENCY))

        price = float(bars[-1].close)
        atr = float(last_row["atr_14_norm"]) * price
        if atr <= 0:
            return

        stop_distance = self._config.stop_atr_mult * atr
        risk_amount = equity * self._config.risk_per_trade_pct * self._regime_kelly * 2
        shares = int(risk_amount / stop_distance)
        max_shares = int(equity * self._config.max_position_pct / price)
        shares = min(shares, max_shares)
        if shares < 1:
            return

        instrument = self.cache.instrument(instrument_id)
        if instrument is None:
            return

        order = self.order_factory.limit(
            instrument_id=instrument_id,
            order_side=OrderSide.BUY,
            quantity=Quantity.from_int(shares),
            price=Price.from_str(f"{price:.2f}"),
            time_in_force=TimeInForce.DAY,
        )
        self.submit_order(order)
        self._entry_bars[sym] = self._total_bars
        self._stop_loss[sym] = price - stop_distance
        self.log.info(
            f"ML BUY {shares} {sym} @ ${price:.2f} "
            f"(conf={confidence:.2f}, stop=${self._stop_loss[sym]:.2f})"
        )

    def _check_time_exit(self, sym: str) -> None:
        instrument_id = InstrumentId.from_str(f"{sym}.ALPACA")
        if not self.portfolio.is_net_long(instrument_id):
            return
        entry_bar = self._entry_bars.get(sym, 0)
        if self._total_bars - entry_bar >= self._config.max_hold_days:
            self.close_all_positions(instrument_id)
            self.log.info(f"ML EXIT {sym}: max hold {self._config.max_hold_days}d")
            self._entry_bars.pop(sym, None)
            self._stop_loss.pop(sym, None)

    def _check_stop_loss(self, sym: str, current_price: float) -> None:
        stop = self._stop_loss.get(sym)
        if stop is None:
            return
        if current_price <= stop:
            instrument_id = InstrumentId.from_str(f"{sym}.ALPACA")
            for position in self.cache.positions_open(strategy_id=self.id):
                if position.instrument_id == instrument_id:
                    self.close_position(position)
                    self.log.warning(
                        f"ML STOP LOSS {sym} @ ${current_price:.2f} (stop=${stop:.2f})"
                    )
                    self._stop_loss.pop(sym, None)
                    self._entry_bars.pop(sym, None)
                    return

    def on_stop(self) -> None:
        for sym in self._config.symbols:
            instrument_id = InstrumentId.from_str(f"{sym}.ALPACA")
            self.cancel_all_orders(instrument_id)
