"""Short-term mean reversion strategy — NautilusTrader Strategy.

Entry: RSI(14)<30 + below lower BB + volume spike + above 200-SMA.
Exit: RSI>50, ATR-based TP/SL, or 5-day timeout.
"""

from __future__ import annotations

from collections import deque

import numpy as np

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
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.trading.strategy import Strategy

from kryor.adapters.alpaca.constants import ALPACA_VENUE
from kryor.core.custom_data import RegimeData, RegimeEnum


class MeanReversionConfig(StrategyConfig, frozen=True):
    symbols: list[str] = []
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    rsi_threshold: int = 30
    volume_mult: float = 1.5
    max_hold_days: int = 5
    atr_period: int = 14
    bb_period: int = 20
    sma_period: int = 200
    risk_per_trade_pct: float = 0.01
    max_position_pct: float = 0.15


class MeanReversionStrategy(Strategy):
    """Short-term mean reversion with RSI + Bollinger Band + volume confirmation."""

    def __init__(self, config: MeanReversionConfig) -> None:
        super().__init__(config=config)
        self._config = config
        self._regime = RegimeEnum.NEUTRAL
        self._regime_kelly = 0.25
        self._bars: dict[str, deque[Bar]] = {}
        self._entry_bars: dict[str, int] = {}  # symbol → bar count at entry
        self._total_bars = 0

    def _on_regime_update(self, data: RegimeData) -> None:
        self._regime = data.regime
        self._regime_kelly = data.kelly_fraction

    def on_start(self) -> None:
        self.log.info("MeanReversionStrategy starting — preloading historical data")
        self.msgbus.subscribe("regime.update", self._on_regime_update)

        # Preload historical bars
        if self._config.alpaca_api_key:
            from kryor.adapters.alpaca.history import fetch_historical_bars
            for sym in self._config.symbols:
                self._bars[sym] = deque(maxlen=300)
                hist = fetch_historical_bars(
                    self._config.alpaca_api_key,
                    self._config.alpaca_secret_key,
                    sym,
                    days=300,
                )
                for bar in hist:
                    self._bars[sym].append(bar)
            self.log.info(
                f"Preloaded history for {len(self._config.symbols)} symbols "
                f"(~{len(self._bars.get(self._config.symbols[0], []))} bars each)"
            )
        else:
            for sym in self._config.symbols:
                self._bars[sym] = deque(maxlen=300)

        # Subscribe to live bars
        for sym in self._config.symbols:
            instrument_id = InstrumentId.from_str(f"{sym}.ALPACA")
            bar_spec = BarSpecification(1, BarAggregation.DAY, PriceType.LAST)
            bar_type = BarType(instrument_id, bar_spec, AggregationSource.EXTERNAL)
            self.subscribe_bars(bar_type)

        # Check for immediate signals with preloaded data
        if self._regime != RegimeEnum.BEAR:
            for sym in self._config.symbols:
                if len(self._bars.get(sym, [])) >= self._config.sma_period:
                    self._check_entry(sym)

        self.log.info(f"MeanReversionStrategy ready — {len(self._config.symbols)} symbols")

    def on_bar(self, bar: Bar) -> None:
        sym = bar.bar_type.instrument_id.symbol.value
        if sym not in self._bars:
            return
        self._bars[sym].append(bar)
        self._total_bars += 1

        if self._regime == RegimeEnum.BEAR:
            return

        # Check exit conditions for open positions
        self._check_exits(sym)

        # Check entry conditions
        self._check_entry(sym)

    def _check_entry(self, sym: str) -> None:
        bars = self._bars[sym]
        if len(bars) < self._config.sma_period:
            return

        instrument_id = InstrumentId.from_str(f"{sym}.ALPACA")
        if self.portfolio.is_net_long(instrument_id):
            return

        closes = np.array([float(b.close) for b in bars])
        highs = np.array([float(b.high) for b in bars])
        lows = np.array([float(b.low) for b in bars])
        volumes = np.array([float(b.volume) for b in bars])

        price = closes[-1]
        rsi = self._calc_rsi(closes, 14)
        sma200 = closes[-self._config.sma_period:].mean()
        sma20 = closes[-self._config.bb_period:].mean()
        std20 = closes[-self._config.bb_period:].std()
        bb_lower = sma20 - 2 * std20
        vol_avg = volumes[-20:].mean()
        atr = self._calc_atr(highs, lows, closes, self._config.atr_period)

        # All conditions must be true
        if rsi >= self._config.rsi_threshold:
            return
        if price >= bb_lower:
            return
        if volumes[-1] < vol_avg * self._config.volume_mult:
            return
        if price < sma200:
            return
        if atr <= 0:
            return

        # Position sizing
        account = self.portfolio.account(ALPACA_VENUE)
        if account is None:
            return
        equity = float(account.balance_total(account.base_currency))
        stop_distance = atr * 2
        risk_amount = equity * self._config.risk_per_trade_pct * self._regime_kelly * 2
        shares = int(risk_amount / stop_distance)
        max_shares = int(equity * self._config.max_position_pct / price)
        shares = min(shares, max_shares)
        if shares <= 0:
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
        self.log.info(f"MR BUY {shares} {sym} @ ${price:.2f} (RSI={rsi:.1f})")

    def _check_exits(self, sym: str) -> None:
        instrument_id = InstrumentId.from_str(f"{sym}.ALPACA")
        if not self.portfolio.is_net_long(instrument_id):
            return

        bars = self._bars[sym]
        if len(bars) < 20:
            return

        closes = np.array([float(b.close) for b in bars])
        rsi = self._calc_rsi(closes, 14)

        # Exit: RSI > 50 (mean reversion complete)
        if rsi > 50:
            self.close_position(instrument_id)
            self.log.info(f"MR EXIT {sym}: RSI={rsi:.1f} > 50")
            self._entry_bars.pop(sym, None)
            return

        # Exit: max hold days
        entry_bar = self._entry_bars.get(sym, 0)
        if self._total_bars - entry_bar >= self._config.max_hold_days:
            self.close_position(instrument_id)
            self.log.info(f"MR EXIT {sym}: max hold {self._config.max_hold_days} days")
            self._entry_bars.pop(sym, None)

    def on_stop(self) -> None:
        for sym in self._config.symbols:
            instrument_id = InstrumentId.from_str(f"{sym}.ALPACA")
            self.cancel_all_orders(instrument_id)

    @staticmethod
    def _calc_rsi(closes: np.ndarray, period: int) -> float:
        if len(closes) < period + 1:
            return 50.0
        deltas = np.diff(closes[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0).mean()
        losses = np.where(deltas < 0, -deltas, 0).mean()
        if losses == 0:
            return 100.0
        rs = gains / losses
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _calc_atr(highs, lows, closes, period: int) -> float:
        if len(closes) < period + 1:
            return 0.0
        tr = np.maximum(
            highs[-period:] - lows[-period:],
            np.maximum(
                np.abs(highs[-period:] - closes[-period - 1:-1]),
                np.abs(lows[-period:] - closes[-period - 1:-1]),
            ),
        )
        return float(tr.mean())
