"""Risk-adjusted momentum strategy — NautilusTrader Strategy.

Features:
  [1] 12-1 month momentum with 200-day SMA trend filter
  [2] Correlation-aware selection (avoids picking highly correlated stocks)
  [3] Risk parity allocation (inverse volatility weights)
  [4] Dynamic rebalancing (time + volatility regime + portfolio drift)
  [5] Stop loss (2*ATR) + BEAR regime forced close
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
from nautilus_trader.model.objects import Currency, Price, Quantity
from nautilus_trader.trading.strategy import Strategy

from kryor.adapters.alpaca.constants import ALPACA_VENUE
from kryor.core.custom_data import RegimeData, RegimeEnum

_USD_CURRENCY = Currency.from_str("USD")


class MomentumConfig(StrategyConfig, frozen=True):
    symbols: list[str] = []
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    # Core momentum
    lookback_days: int = 252
    skip_days: int = 21
    top_pct: float = 0.10
    atr_period: int = 14
    sma_period: int = 200
    # Position sizing
    target_portfolio_risk: float = 0.02  # 総ポートフォリオリスク2% (Risk parity basis)
    max_position_pct: float = 0.15
    # Dynamic rebalancing
    rebalance_interval_days: int = 21
    min_rebalance_interval_days: int = 10  # 最低間隔（ドリフト/ボラ起因でも）
    vol_spike_threshold: float = 2.0  # ATR平均が2倍になったら即リバランス
    drift_threshold: float = 0.75  # ポジションウェイトが目標から75%ずれたらリバランス
    # Correlation
    correlation_window: int = 60  # 相関計算のルックバック
    max_correlation: float = 0.70  # この閾値以上の相関ペアは同時保有しない


class MomentumStrategy(Strategy):
    """Portfolio-theory-aware momentum strategy."""

    def __init__(self, config: MomentumConfig) -> None:
        super().__init__(config=config)
        self._config = config
        self._regime = RegimeEnum.NEUTRAL
        self._regime_kelly = 0.25
        self._bars: dict[str, deque[Bar]] = {}
        self._bar_count = 0
        self._stop_loss: dict[str, float] = {}
        self._target_weights: dict[str, float] = {}  # Risk parity target weights
        self._last_rebalance_bar: int = 0
        self._last_vol_avg: float | None = None
        self._cb_level: int = 0  # 0 = OK, 4 = halted

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
        if data.level >= 3:  # Weekly DD or higher → close everything
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
        self.log.info("MomentumStrategy starting — preloading historical data")
        self.msgbus.subscribe("regime.update", self._on_regime_update)
        self.msgbus.subscribe("circuit_breaker.update", self._on_circuit_breaker)

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

        for sym in self._config.symbols:
            instrument_id = InstrumentId.from_str(f"{sym}.ALPACA")
            bar_spec = BarSpecification(1, BarAggregation.DAY, PriceType.LAST)
            bar_type = BarType(instrument_id, bar_spec, AggregationSource.EXTERNAL)
            self.subscribe_bars(bar_type)

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

        # [1] ストップロス（毎バー）
        self._check_stop_loss(sym, float(bar.close))

        # [2] 日次カウンタはSPYバーのみで進める（全銘柄で加算すると36倍になる）
        if sym != "SPY" and sym != self._config.symbols[0]:
            return
        self._bar_count += 1

        if self._regime == RegimeEnum.BEAR:
            return

        # CB Level 2以上なら新規エントリー停止
        if self._cb_level >= 2:
            return

        # [3] 動的リバランス判定
        if self._should_rebalance():
            self._rebalance()
            self._last_rebalance_bar = self._bar_count

    def _should_rebalance(self) -> bool:
        """月次の固定リバランス + 大きなボラスパイク時の追加リバランス"""
        bars_since_rebalance = self._bar_count - self._last_rebalance_bar

        # Trigger 1: 時間経過（21営業日）
        if bars_since_rebalance >= self._config.rebalance_interval_days:
            return True

        # 最低間隔は厳守
        if bars_since_rebalance < self._config.min_rebalance_interval_days:
            return False

        # Trigger 2: 大きなボラスパイク時のみ
        current_vol_avg = self._compute_avg_atr()
        if self._last_vol_avg and current_vol_avg:
            if current_vol_avg > self._last_vol_avg * self._config.vol_spike_threshold:
                self.log.info(
                    f"Vol spike: {current_vol_avg:.3f} > {self._last_vol_avg:.3f} "
                    f"× {self._config.vol_spike_threshold}"
                )
                return True

        return False

    def _compute_avg_atr(self) -> float:
        """ポートフォリオ内の保有銘柄の平均ATR/価格比"""
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

    def _check_drift(self) -> bool:
        """ポジションの現在ウェイトが目標から drift_threshold 以上乖離したか"""
        account = self.portfolio.account(ALPACA_VENUE)
        if account is None:
            return False
        equity = float(account.balance_total(account.base_currency or _USD_CURRENCY))
        if equity <= 0:
            return False

        for position in self.cache.positions_open(strategy_id=self.id):
            sym = position.instrument_id.symbol.value
            target_w = self._target_weights.get(sym, 0)
            if target_w == 0:
                continue
            bars = self._bars.get(sym, [])
            if not bars:
                continue
            current_price = float(bars[-1].close)
            current_w = (float(position.quantity) * current_price) / equity
            drift = abs(current_w - target_w) / target_w
            if drift > self._config.drift_threshold:
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
        # [1] スコアリング
        scores: dict[str, float] = {}
        vols: dict[str, float] = {}

        for sym, bars in self._bars.items():
            if len(bars) < self._config.lookback_days:
                continue
            closes = np.array([float(b.close) for b in bars])
            highs = np.array([float(b.high) for b in bars])
            lows = np.array([float(b.low) for b in bars])

            # トレンドフィルター
            if len(closes) >= self._config.sma_period:
                sma200 = closes[-self._config.sma_period:].mean()
                if closes[-1] < sma200:
                    continue

            # 12-1 momentum
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

        # [2] 相関ベースの銘柄選択
        selected = self._correlation_aware_selection(scores, vols)
        if not selected:
            return

        # [3] Risk Parity でウェイト計算
        self._target_weights = self._risk_parity_weights(selected, vols)

        # [4] 選外ポジション決済
        selected_set = set(selected)
        for position in self.cache.positions_open(strategy_id=self.id):
            sym = position.instrument_id.symbol.value
            if sym not in selected_set:
                self.close_position(position)
                self._stop_loss.pop(sym, None)

        # [5] 新規エントリー/リサイズ
        self._execute_target_weights()

        self._last_vol_avg = self._compute_avg_atr()

    def _correlation_aware_selection(
        self, scores: dict[str, float], vols: dict[str, float]
    ) -> list[str]:
        """スコア順で選ぶが、既選銘柄と相関0.7以上のものはスキップ"""
        sorted_syms = sorted(scores, key=lambda s: scores[s], reverse=True)
        n_target = max(1, int(len(sorted_syms) * self._config.top_pct))

        # 直近N日のリターンで相関を計算
        returns: dict[str, np.ndarray] = {}
        w = self._config.correlation_window
        for sym in sorted_syms:
            bars_list = list(self._bars[sym])[-(w + 1):]
            closes = np.array([float(b.close) for b in bars_list])
            if len(closes) >= w + 1:
                returns[sym] = np.diff(closes) / closes[:-1]

        selected: list[str] = []
        for sym in sorted_syms:
            if len(selected) >= n_target:
                break
            if sym not in returns:
                selected.append(sym)
                continue

            # 既選銘柄との最大相関を確認
            too_correlated = False
            for existing in selected:
                if existing not in returns:
                    continue
                try:
                    corr = np.corrcoef(returns[sym], returns[existing])[0, 1]
                    if not np.isnan(corr) and abs(corr) > self._config.max_correlation:
                        too_correlated = True
                        break
                except Exception:
                    pass
            if not too_correlated:
                selected.append(sym)

        if len(selected) < n_target:
            self.log.info(
                f"Correlation filter: selected {len(selected)}/{n_target} "
                f"(rejected due to corr > {self._config.max_correlation})"
            )
        return selected

    def _risk_parity_weights(
        self, symbols: list[str], vols: dict[str, float]
    ) -> dict[str, float]:
        """各銘柄が同じリスクを負担するよう逆ボラウェイト"""
        inv_vols = {s: 1.0 / vols[s] for s in symbols if s in vols and vols[s] > 0}
        total = sum(inv_vols.values())
        if total <= 0:
            return {}
        # レジーム調整: Kelly分率でポートフォリオ総エクスポージャを調整
        scale = self._regime_kelly * 2  # 0.5→1.0, 0.25→0.5
        return {s: (inv_v / total) * scale for s, inv_v in inv_vols.items()}

    def _execute_target_weights(self) -> None:
        """新規エントリーのみ。既存ポジションはサイズ変更しない（高値掴み回避）。"""
        account = self.portfolio.account(ALPACA_VENUE)
        if account is None:
            return
        equity = float(account.balance_total(account.base_currency or _USD_CURRENCY))

        # 既存ポジション銘柄を取得
        existing_symbols = {p.instrument_id.symbol.value
                           for p in self.cache.positions_open(strategy_id=self.id)}

        for sym, target_w in self._target_weights.items():
            # 既存ポジションがある銘柄はサイズ調整しない
            if sym in existing_symbols:
                continue

            bars = self._bars[sym]
            closes = np.array([float(b.close) for b in bars])
            highs = np.array([float(b.high) for b in bars])
            lows = np.array([float(b.low) for b in bars])
            price = closes[-1]
            atr = self._calc_atr(highs, lows, closes, self._config.atr_period)
            if atr <= 0:
                continue

            # 目標ウェイトから株数を計算（新規エントリー時のみ）
            target_value = equity * target_w
            target_shares = int(target_value / price)

            # max_position_pct制約
            max_shares = int(equity * self._config.max_position_pct / price)
            target_shares = min(target_shares, max_shares)
            if target_shares < 1:
                continue

            instrument_id = InstrumentId.from_str(f"{sym}.ALPACA")
            instrument = self.cache.instrument(instrument_id)
            if instrument is None:
                continue

            order = self.order_factory.limit(
                instrument_id=instrument_id,
                order_side=OrderSide.BUY,
                quantity=Quantity.from_int(target_shares),
                price=Price.from_str(f"{price:.2f}"),
                time_in_force=TimeInForce.DAY,
            )
            self.submit_order(order)
            stop_price = price - 2 * atr
            self._stop_loss[sym] = stop_price
            self.log.info(
                f"BUY {target_shares} {sym} @ ${price:.2f} "
                f"(target_w={target_w:.1%}, vol={atr/price:.2%}, stop=${stop_price:.2f})"
            )

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
