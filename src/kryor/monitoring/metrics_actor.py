"""Prometheus MetricsActor — NT Actor that exposes portfolio/execution/regime metrics.

Listens to NT events (fills, positions, account updates) and regime/CB data
via MessageBus, then updates Prometheus gauges/counters/histograms.
Prometheus scrapes :8000/metrics, Grafana reads from Prometheus.
"""

from __future__ import annotations

import time
from collections import deque

from prometheus_client import Counter, Gauge, Histogram, Info, start_http_server

from nautilus_trader.common.actor import Actor
from nautilus_trader.config import ActorConfig
from nautilus_trader.model.events import (
    OrderFilled,
    PositionChanged,
    PositionClosed,
    PositionOpened,
    AccountState,
)

from kryor.adapters.alpaca.constants import ALPACA_VENUE
from kryor.core.custom_data import CircuitBreakerData, RegimeData


# ── Prometheus Metrics ──────────────────────────────────────

# Account / P&L
equity = Gauge("kryor_equity_usd", "Current account equity in USD")
cash = Gauge("kryor_cash_usd", "Current available cash in USD")
unrealized_pnl = Gauge("kryor_unrealized_pnl_usd", "Total unrealized P&L")
realized_pnl_total = Counter("kryor_realized_pnl_total_usd", "Cumulative realized P&L")
daily_pnl = Gauge("kryor_daily_pnl_usd", "P&L since start of day")
drawdown = Gauge("kryor_drawdown_pct", "Current drawdown from peak equity")

# Positions
open_positions = Gauge("kryor_open_positions", "Number of open positions")
total_exposure = Gauge("kryor_total_exposure_usd", "Total position market value")

# Orders / Fills
fills_total = Counter("kryor_fills_total", "Total order fills", ["side", "strategy"])
orders_submitted = Counter("kryor_orders_submitted_total", "Total orders submitted", ["side"])
slippage_bps = Histogram(
    "kryor_slippage_bps", "Fill slippage in basis points",
    buckets=[0.5, 1, 2, 5, 10, 20, 50],
)
fill_value = Histogram(
    "kryor_fill_value_usd", "Fill notional value",
    buckets=[100, 500, 1000, 5000, 10000, 50000],
)

# Regime
regime_state = Gauge("kryor_regime", "Current regime: 0=bull, 1=neutral, 2=bear")
regime_probability = Gauge("kryor_regime_probability", "Regime probability")
regime_bull_prob = Gauge("kryor_regime_bull_prob", "Bull probability")
regime_neutral_prob = Gauge("kryor_regime_neutral_prob", "Neutral probability")
regime_bear_prob = Gauge("kryor_regime_bear_prob", "Bear probability")

# Circuit Breaker
cb_level = Gauge("kryor_circuit_breaker_level", "Active circuit breaker level (0=none)")

# System
uptime_seconds = Gauge("kryor_uptime_seconds", "Engine uptime in seconds")
engine_info = Info("kryor_engine", "Engine metadata")


class MetricsActorConfig(ActorConfig, frozen=True):
    prometheus_port: int = 8000
    update_interval_secs: float = 10.0


class MetricsActor(Actor):
    """NT Actor that collects trading metrics and exposes them to Prometheus."""

    def __init__(self, config: MetricsActorConfig) -> None:
        super().__init__(config=config)
        self._config = config
        self._start_time: float = 0
        self._peak_equity: float = 0
        self._day_start_equity: float = 0
        self._daily_returns: deque[float] = deque(maxlen=30)
        self._last_equity: float = 0

    def on_start(self) -> None:
        self._start_time = time.time()
        engine_info.info({
            "trader_id": "KRYOR-001",
            "version": "0.1.0",
            "mode": "paper",
        })

        # Subscribe to regime and CB updates via msgbus
        self.msgbus.subscribe("regime.update", self._on_regime)
        self.msgbus.subscribe("circuit_breaker.update", self._on_cb)

        # Schedule periodic portfolio sync
        from datetime import timedelta
        self.clock.set_timer(
            name="metrics_sync",
            interval=timedelta(seconds=self._config.update_interval_secs),
            callback=self._on_timer,
        )

        # Initialize all metrics with defaults so they appear in Prometheus
        regime_state.set(1)  # NEUTRAL
        regime_probability.set(0.5)
        regime_bull_prob.set(0.25)
        regime_neutral_prob.set(0.50)
        regime_bear_prob.set(0.25)
        cb_level.set(0)
        open_positions.set(0)
        total_exposure.set(0)
        unrealized_pnl.set(0)
        daily_pnl.set(0)
        drawdown.set(0)
        uptime_seconds.set(0)

        # Initialize equity
        self._sync_account()
        self.log.info("MetricsActor started")

    def on_event(self, event) -> None:
        if isinstance(event, OrderFilled):
            self._on_fill(event)
        elif isinstance(event, (PositionOpened, PositionChanged, PositionClosed)):
            self._on_position_event(event)
        elif isinstance(event, AccountState):
            self._on_account_state(event)

    def _on_fill(self, event: OrderFilled) -> None:
        side = "buy" if "BUY" in str(event.order_side) else "sell"
        strategy = str(event.strategy_id) if event.strategy_id else "unknown"
        fills_total.labels(side=side, strategy=strategy).inc()

        notional = float(event.last_qty) * float(event.last_px)
        fill_value.observe(notional)

        self.log.info(
            f"FILL: {side} {event.last_qty} @ {event.last_px} "
            f"(strategy={strategy}, value=${notional:,.2f})"
        )

    def _on_position_event(self, event) -> None:
        self._sync_positions()

    def _on_account_state(self, event: AccountState) -> None:
        for balance in event.balances:
            if str(balance.currency) == "USD":
                eq = float(balance.total)
                equity.set(eq)
                cash.set(float(balance.free))

                if eq > self._peak_equity:
                    self._peak_equity = eq
                if self._peak_equity > 0:
                    dd = (self._peak_equity - eq) / self._peak_equity
                    drawdown.set(dd)

                if self._day_start_equity > 0:
                    daily_pnl.set(eq - self._day_start_equity)
                else:
                    self._day_start_equity = eq

                self._last_equity = eq

    def _on_regime(self, data: RegimeData) -> None:
        regime_state.set(int(data.regime))
        regime_probability.set(data.probability)
        regime_bull_prob.set(data.bull_prob)
        regime_neutral_prob.set(data.neutral_prob)
        regime_bear_prob.set(data.bear_prob)

    def _on_cb(self, data: CircuitBreakerData) -> None:
        cb_level.set(data.level)
        self.log.warning(f"CB metric updated: level={data.level} reason={data.reason}")

    def _on_timer(self, event) -> None:
        uptime_seconds.set(time.time() - self._start_time)
        self._sync_account()
        self._sync_positions()

    def _sync_account(self) -> None:
        account = self.portfolio.account(ALPACA_VENUE)
        if account is None:
            return
        eq = float(account.balance_total(account.base_currency))
        equity.set(eq)

        if eq > self._peak_equity:
            self._peak_equity = eq
        if self._peak_equity > 0:
            drawdown.set((self._peak_equity - eq) / self._peak_equity)

        if self._day_start_equity == 0:
            self._day_start_equity = eq
        daily_pnl.set(eq - self._day_start_equity)
        self._last_equity = eq

    def _sync_positions(self) -> None:
        positions = self.cache.positions_open()
        open_positions.set(len(positions))

        exposure = 0.0
        pnl = 0.0
        for pos in positions:
            exposure += abs(float(pos.quantity) * float(pos.avg_px_open))
            pnl += float(pos.unrealized_pnl) if pos.unrealized_pnl else 0

        total_exposure.set(exposure)
        unrealized_pnl.set(pnl)
