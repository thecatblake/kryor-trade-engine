"""Risk-adjusted momentum strategy — NautilusTrader Strategy.

Features:
  - 12-1 month momentum with 200-day SMA trend filter
  - PortfolioConfig-driven sizing (scales with capital)
  - Correlation-aware selection + sector limits
  - Risk parity allocation (inverse volatility weights)
  - Fractional shares support
  - Stop loss (2*ATR) + BEAR regime forced close
"""

from __future__ import annotations

import math
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
from nautilus_trader.model.objects import Currency, Price, Quantity
from nautilus_trader.trading.strategy import Strategy

from kryor.adapters.alpaca.constants import ALPACA_VENUE
from kryor.core.custom_data import RegimeData, RegimeEnum
from kryor.core.portfolio_config import PortfolioConfig, select_universe

_USD_CURRENCY = Currency.from_str("USD")


class MomentumConfig(StrategyConfig, frozen=True):
    symbols: list[str] = []
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    # Core momentum
    lookback_days: int = 252
    skip_days: int = 21
    atr_period: int = 14
    sma_period: int = 200
    # Dynamic rebalancing
    min_rebalance_interval_days: int = 10
    vol_spike_threshold: float = 2.0
    # Correlation
    correlation_window: int = 60
    max_correlation: float = 0.70


class MomentumStrategy(Strategy):
    """Portfolio-theory-aware momentum strategy with capital-adaptive sizing."""

    def __init__(self, config: MomentumConfig) -> None:
        super().__init__(config=config)
        self._config = config
        self._regime = RegimeEnum.NEUTRAL
        self._regime_kelly = 0.25
        self._cb_level = 0
        self._bars: dict[str, deque[Bar]] = {}
        self._bar_count = 0
        self._stop_loss: dict[str, float] = {}
        self._target_weights: dict[str, float] = {}
        self._last_rebalance_bar: int = 0
        self._last_vol_avg: float | None = None
        self._pf_config: PortfolioConfig | None = None

    def _on_regime_update(self, data: RegimeData) -> None:
        prev = self._regime
        self._regime = data.regime
        self._regime_kelly = data.kelly_fraction
        self.log.info(f"Regime updated: {data.regime.name} (kelly={data.kelly_fraction})")
        if data.regime == RegimeEnum.BEAR and prev != RegimeEnum.BEAR:
            self._close_all_positions("BEAR regime")

    def _on_circuit_breaker(self, data) -> None:
        self._cb_level = data.level
        self.log.warning(f"CB L{data.level} received: {data.reason}")
        if data.level >= 3:
            self._close_all_positions(f"CB L{data.level}: {data.reason}")

    def _close_all_positions(self, reason: str) -> None:
        closed = 0
        for position in self.cache.positions_open(strategy_id=self.id):
            self.close_position(position)
            closed += 1
        self._stop_loss.clear()
        self._target_weights.clear()
        if closed > 0:
            self.log.warning(f"Closed {closed} positions: {reason}")

    def on_start(self) -> None:
        self.log.info("MomentumStrategy starting")
        self.msgbus.subscribe("regime.update", self._on_regime_update)
        self.msgbus.subscribe("circuit_breaker.update", self._on_circuit_breaker)

        # Preload
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
                f"Preloaded ~{len(self._bars.get(self._config.symbols[0], []))} bars/symbol"
            )
        else:
            for sym in self._config.symbols:
                self._bars[sym] = deque(maxlen=self._config.lookback_days + 50)

        for sym in self._config.symbols:
            instrument_id = InstrumentId.from_str(f"{sym}.ALPACA")
            bar_spec = BarSpecification(1, BarAggregation.DAY, PriceType.LAST)
            bar_type = BarType(instrument_id, bar_spec, AggregationSource.EXTERNAL)
            self.subscribe_bars(bar_type)

        # Initial rebalance
        self._refresh_portfolio_config()
        if self._regime != RegimeEnum.BEAR and self._pf_config:
            first_sym = self._config.symbols[0]
            if len(self._bars.get(first_sym, [])) >= self._config.lookback_days:
                self.log.info("Running initial rebalance")
                self._rebalance()

        self.log.info(f"MomentumStrategy ready — {len(self._config.symbols)} symbols")

    def on_bar(self, bar: Bar) -> None:
        sym = bar.bar_type.instrument_id.symbol.value
        if sym not in self._bars:
            return
        self._bars[sym].append(bar)

        # Stop loss
        self._check_stop_loss(sym, float(bar.close))

        # Day counter on SPY or first symbol only
        if sym != "SPY" and sym != self._config.symbols[0]:
            return
        self._bar_count += 1

        if self._regime == RegimeEnum.BEAR or self._cb_level >= 2:
            return

        if self._should_rebalance():
            self._refresh_portfolio_config()
            self._rebalance()
            self._last_rebalance_bar = self._bar_count

    def _refresh_portfolio_config(self) -> None:
        """現在のエクイティからPortfolioConfigを再計算"""
        account = self.portfolio.account(ALPACA_VENUE)
        if account is None:
            return
        equity = float(account.balance_total(account.base_currency or _USD_CURRENCY))
        self._pf_config = PortfolioConfig.from_equity(equity)
        self.log.info(f"Portfolio config refreshed:\n{self._pf_config.describe()}")

    def _should_rebalance(self) -> bool:
        if self._pf_config is None:
            return False
        bars_since = self._bar_count - self._last_rebalance_bar
        if bars_since >= self._pf_config.rebalance_interval_days:
            return True
        if bars_since < self._config.min_rebalance_interval_days:
            return False
        current_vol = self._compute_avg_atr()
        if self._last_vol_avg and current_vol:
            if current_vol > self._last_vol_avg * self._config.vol_spike_threshold:
                self.log.info(f"Vol spike: {current_vol:.4f} > {self._last_vol_avg:.4f} × {self._config.vol_spike_threshold}")
                return True
        return False

    def _check_stop_loss(self, sym: str, current_price: float) -> None:
        stop = self._stop_loss.get(sym)
        if stop is None:
            return
        if current_price <= stop:
            instrument_id = InstrumentId.from_str(f"{sym}.ALPACA")
            for position in self.cache.positions_open(strategy_id=self.id):
                if position.instrument_id == instrument_id:
                    self.close_position(position)
                    self.log.warning(f"STOP LOSS {sym} @ ${current_price:.2f} (stop=${stop:.2f})")
                    self._stop_loss.pop(sym, None)
                    self._target_weights.pop(sym, None)
                    return

    # ── Rebalancing ────────────────────────────────────────

    def _rebalance(self) -> None:
        if self._pf_config is None:
            return

        # [1] Score all symbols
        scores: dict[str, float] = {}
        vols: dict[str, float] = {}
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
            vol_ratio = atr / closes[-1]
            scores[sym] = raw_momentum / vol_ratio
            vols[sym] = vol_ratio

        if not scores:
            return

        # [1.5] Price filter — 高すぎる銘柄を除外
        max_price = self._pf_config.max_stock_price
        filtered_scores = {}
        for sym, score in scores.items():
            bars = self._bars[sym]
            price = float(bars[-1].close) if bars else 0
            if 0 < price <= max_price:
                filtered_scores[sym] = score
        if filtered_scores:
            removed = len(scores) - len(filtered_scores)
            if removed > 0:
                self.log.info(f"Price filter: removed {removed} symbols > ${max_price:.0f}")
            scores = filtered_scores
        if not scores:
            return

        # [2] Correlation matrix
        correlations = self._compute_correlations(list(scores.keys()))

        # [3] Select universe (PortfolioConfig-driven)
        selected = select_universe(
            candidates=list(scores.keys()),
            scores=scores,
            config=self._pf_config,
            correlations=correlations,
            max_correlation=self._config.max_correlation,
        )
        if not selected:
            return

        self.log.info(
            f"Selected {len(selected)}/{len(scores)} symbols: {selected} "
            f"(max_positions={self._pf_config.max_positions})"
        )

        # [4] Risk Parity weights
        self._target_weights = self._risk_parity_weights(selected, vols)

        # [5] Close positions not in selected
        for position in self.cache.positions_open(strategy_id=self.id):
            sym = position.instrument_id.symbol.value
            if sym not in set(selected):
                self.close_position(position)
                self._stop_loss.pop(sym, None)

        # [6] Execute
        self._execute_target_weights()
        self._last_vol_avg = self._compute_avg_atr()

    def _compute_correlations(self, symbols: list[str]) -> dict[tuple[str, str], float]:
        """直近N日のリターン相関を計算"""
        w = self._config.correlation_window
        returns: dict[str, np.ndarray] = {}
        for sym in symbols:
            bars_list = list(self._bars[sym])[-(w + 1):]
            closes = np.array([float(b.close) for b in bars_list])
            if len(closes) >= w + 1:
                returns[sym] = np.diff(closes) / closes[:-1]

        correlations = {}
        syms = list(returns.keys())
        for i in range(len(syms)):
            for j in range(i + 1, len(syms)):
                try:
                    corr = float(np.corrcoef(returns[syms[i]], returns[syms[j]])[0, 1])
                    if not np.isnan(corr):
                        key = (min(syms[i], syms[j]), max(syms[i], syms[j]))
                        correlations[key] = corr
                except Exception:
                    pass
        return correlations

    def _risk_parity_weights(
        self, symbols: list[str], vols: dict[str, float]
    ) -> dict[str, float]:
        inv_vols = {s: 1.0 / vols[s] for s in symbols if s in vols and vols[s] > 0}
        total = sum(inv_vols.values())
        if total <= 0:
            return {}
        # Scale by regime Kelly + cash reserve
        cash_adj = 1 - (self._pf_config.cash_reserve_pct if self._pf_config else 0.2)
        scale = self._regime_kelly * 2 * cash_adj
        weights = {s: (inv_v / total) * scale for s, inv_v in inv_vols.items()}
        # Cap at max_position_pct
        max_w = self._pf_config.max_position_pct if self._pf_config else 0.15
        return {s: min(w, max_w) for s, w in weights.items()}

    def _execute_target_weights(self) -> None:
        account = self.portfolio.account(ALPACA_VENUE)
        if account is None:
            return
        equity = float(account.balance_total(account.base_currency or _USD_CURRENCY))

        existing = {p.instrument_id.symbol.value
                    for p in self.cache.positions_open(strategy_id=self.id)}

        use_frac = self._pf_config.use_fractional if self._pf_config else False
        min_pos = self._pf_config.min_position_usd if self._pf_config else 100

        for sym, target_w in self._target_weights.items():
            if sym in existing:
                continue

            bars = self._bars[sym]
            closes = np.array([float(b.close) for b in bars])
            highs = np.array([float(b.high) for b in bars])
            lows = np.array([float(b.low) for b in bars])
            price = closes[-1]
            atr = self._calc_atr(highs, lows, closes, self._config.atr_period)
            if atr <= 0:
                continue

            target_value = equity * target_w
            if target_value < min_pos:
                continue

            # 端株 or 整数株
            if use_frac:
                target_qty = round(target_value / price, 4)  # 小数4桁
            else:
                target_qty = int(target_value / price)

            max_pct = self._pf_config.max_position_pct if self._pf_config else 0.15
            max_qty = (equity * max_pct) / price
            if use_frac:
                target_qty = round(min(target_qty, max_qty), 4)
            else:
                target_qty = min(int(target_qty), int(max_qty))

            if target_qty <= 0:
                continue

            instrument_id = InstrumentId.from_str(f"{sym}.ALPACA")
            instrument = self.cache.instrument(instrument_id)
            if instrument is None:
                continue

            qty_str = f"{target_qty:.4f}" if use_frac else str(int(target_qty))
            order = self.order_factory.limit(
                instrument_id=instrument_id,
                order_side=OrderSide.BUY,
                quantity=Quantity.from_str(qty_str),
                price=Price.from_str(f"{price:.2f}"),
                time_in_force=TimeInForce.DAY,
            )
            self.submit_order(order)
            stop_price = price - 2 * atr
            self._stop_loss[sym] = stop_price
            self.log.info(
                f"BUY {qty_str} {sym} @ ${price:.2f} "
                f"(${target_value:.0f}, w={target_w:.1%}, stop=${stop_price:.2f})"
            )

    def _compute_avg_atr(self) -> float:
        open_syms = [p.instrument_id.symbol.value
                     for p in self.cache.positions_open(strategy_id=self.id)]
        if not open_syms:
            return 0.0
        ratios = []
        for sym in open_syms:
            bars = self._bars.get(sym, [])
            if len(bars) < 20:
                continue
            closes = np.array([float(b.close) for b in bars])
            highs = np.array([float(b.high) for b in bars])
            lows = np.array([float(b.low) for b in bars])
            atr = self._calc_atr(highs, lows, closes, self._config.atr_period)
            if atr > 0 and closes[-1] > 0:
                ratios.append(atr / closes[-1])
        return float(np.mean(ratios)) if ratios else 0.0

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
