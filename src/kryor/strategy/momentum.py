"""Risk-adjusted momentum strategy — NautilusTrader Strategy.

12-1 month momentum, vol-adjusted, with 200-day SMA trend filter.
Rebalances monthly. Uses regime data to scale position sizes.
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


class MomentumConfig(StrategyConfig, frozen=True):
    symbols: list[str] = []
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    lookback_days: int = 252
    skip_days: int = 21
    top_pct: float = 0.10
    atr_period: int = 14
    sma_period: int = 200
    risk_per_trade_pct: float = 0.01
    max_position_pct: float = 0.15
    rebalance_interval_days: int = 21


class MomentumStrategy(Strategy):
    """12-1 month risk-adjusted momentum with SMA trend filter."""

    def __init__(self, config: MomentumConfig) -> None:
        super().__init__(config=config)
        self._config = config
        self._regime = RegimeEnum.NEUTRAL
        self._regime_kelly = 0.25
        self._bars: dict[str, deque[Bar]] = {}
        self._bar_count = 0

    def _on_regime_update(self, data: RegimeData) -> None:
        self._regime = data.regime
        self._regime_kelly = data.kelly_fraction
        self.log.info(f"Regime updated: {data.regime.name} (kelly={data.kelly_fraction})")

    def on_start(self) -> None:
        self.log.info("MomentumStrategy starting — preloading historical data")
        self.msgbus.subscribe("regime.update", self._on_regime_update)

        # Preload historical bars so strategy can trade immediately
        if self._config.alpaca_api_key:
            from kryor.adapters.alpaca.history import fetch_historical_bars
            for sym in self._config.symbols:
                self._bars[sym] = deque(maxlen=self._config.lookback_days + 50)
                hist = fetch_historical_bars(
                    self._config.alpaca_api_key,
                    self._config.alpaca_secret_key,
                    sym,
                    days=self._config.lookback_days + 60,
                )
                for bar in hist:
                    self._bars[sym].append(bar)
            self.log.info(
                f"Preloaded history for {len(self._config.symbols)} symbols "
                f"(~{len(self._bars.get(self._config.symbols[0], []))} bars each)"
            )
        else:
            for sym in self._config.symbols:
                self._bars[sym] = deque(maxlen=self._config.lookback_days + 50)

        # Subscribe to live bars
        for sym in self._config.symbols:
            instrument_id = InstrumentId.from_str(f"{sym}.ALPACA")
            bar_spec = BarSpecification(1, BarAggregation.DAY, PriceType.LAST)
            bar_type = BarType(instrument_id, bar_spec, AggregationSource.EXTERNAL)
            self.subscribe_bars(bar_type)

        # Run initial rebalance with preloaded data
        if self._regime != RegimeEnum.BEAR and self._bars:
            first_sym = self._config.symbols[0]
            if len(self._bars.get(first_sym, [])) >= self._config.lookback_days:
                self.log.info("Running initial rebalance with preloaded data")
                self._rebalance()

        self.log.info(f"MomentumStrategy ready — {len(self._config.symbols)} symbols")

    def on_bar(self, bar: Bar) -> None:
        sym = bar.bar_type.instrument_id.symbol.value
        if sym not in self._bars:
            return
        self._bars[sym].append(bar)

        self._bar_count += 1
        if self._bar_count % self._config.rebalance_interval_days != 0:
            return
        if self._regime == RegimeEnum.BEAR:
            self.log.info("Bear regime — skipping rebalance")
            return
        self._rebalance()

    def _rebalance(self) -> None:
        scores: dict[str, float] = {}

        for sym, bars in self._bars.items():
            if len(bars) < self._config.lookback_days:
                continue
            closes = np.array([float(b.close) for b in bars])
            highs = np.array([float(b.high) for b in bars])
            lows = np.array([float(b.low) for b in bars])

            if len(closes) >= self._config.sma_period:
                sma200 = closes[-self._config.sma_period:].mean()
                if closes[-1] < sma200:
                    continue

            total_ret = closes[-1] / closes[-self._config.lookback_days] - 1
            recent_ret = closes[-1] / closes[-self._config.skip_days] - 1
            raw_momentum = total_ret - recent_ret
            atr = self._calc_atr(highs, lows, closes, self._config.atr_period)
            if atr <= 0:
                continue
            scores[sym] = raw_momentum / (atr / closes[-1])

        if not scores:
            return

        sorted_syms = sorted(scores, key=lambda s: scores[s], reverse=True)
        n_select = max(1, int(len(sorted_syms) * self._config.top_pct))
        selected = set(sorted_syms[:n_select])

        # Close positions not in selected
        for position in self.cache.positions_open():
            sym = position.instrument_id.symbol.value
            if sym not in selected:
                self.close_position(position.instrument_id)

        # Open new positions
        account = self.portfolio.account(ALPACA_VENUE)
        if account is None:
            return
        equity = float(account.balance_total(account.base_currency))

        for sym in selected:
            instrument_id = InstrumentId.from_str(f"{sym}.ALPACA")
            if self.portfolio.is_net_long(instrument_id):
                continue

            bars = self._bars[sym]
            closes = np.array([float(b.close) for b in bars])
            highs = np.array([float(b.high) for b in bars])
            lows = np.array([float(b.low) for b in bars])
            price = closes[-1]
            atr = self._calc_atr(highs, lows, closes, self._config.atr_period)
            if atr <= 0:
                continue

            stop_distance = atr * 2
            risk_amount = equity * self._config.risk_per_trade_pct * self._regime_kelly * 2
            shares = int(risk_amount / stop_distance)
            max_shares = int(equity * self._config.max_position_pct / price)
            shares = min(shares, max_shares)
            if shares <= 0:
                continue

            instrument = self.cache.instrument(instrument_id)
            if instrument is None:
                continue

            order = self.order_factory.limit(
                instrument_id=instrument_id,
                order_side=OrderSide.BUY,
                quantity=Quantity.from_int(shares),
                price=Price.from_str(f"{price:.2f}"),
                time_in_force=TimeInForce.DAY,
            )
            self.submit_order(order)
            self.log.info(f"BUY {shares} {sym} @ ${price:.2f} (score={scores[sym]:.3f})")

    def on_stop(self) -> None:
        for sym in self._config.symbols:
            instrument_id = InstrumentId.from_str(f"{sym}.ALPACA")
            self.cancel_all_orders(instrument_id)

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
