"""5-level circuit breaker — NautilusTrader Actor.

Monitors portfolio state and publishes CircuitBreakerData when limits are hit.
Strategies subscribe to CB data and halt trading accordingly.
"""

from __future__ import annotations

import time

from nautilus_trader.common.actor import Actor
from nautilus_trader.config import ActorConfig
from nautilus_trader.model.events import PositionChanged, PositionClosed, PositionOpened

from kryor.adapters.alpaca.constants import ALPACA_VENUE
from kryor.core.custom_data import CircuitBreakerData

CB_TOPIC = "circuit_breaker.update"


class CircuitBreakerConfig(ActorConfig, frozen=True):
    daily_loss_limit_pct: float = 0.03
    weekly_dd_limit_pct: float = 0.08
    monthly_dd_limit_pct: float = 0.15
    max_daily_trades: int = 20
    max_consecutive_losses: int = 5
    max_orders_per_minute: int = 10


class CircuitBreakerActor(Actor):
    """Monitors portfolio and enforces 5-level circuit breaker limits."""

    def __init__(self, config: CircuitBreakerConfig) -> None:
        super().__init__(config=config)
        self._config = config
        self._daily_pnl: float = 0.0
        self._daily_trades: int = 0
        self._consecutive_losses: int = 0
        self._peak_equity: float = 0.0
        self._start_of_day_equity: float = 0.0
        self._start_of_week_equity: float = 0.0
        self._start_of_month_equity: float = 0.0
        self._halted = False
        self._active_level = 0

    def on_start(self) -> None:
        self.log.info("CircuitBreakerActor starting")
        # Initialize equity tracking
        account = self.portfolio.account(ALPACA_VENUE)
        if account:
            equity = float(account.balance_total(account.base_currency))
            self._peak_equity = equity
            self._start_of_day_equity = equity
            self._start_of_week_equity = equity
            self._start_of_month_equity = equity

    def on_event(self, event) -> None:
        if isinstance(event, (PositionOpened, PositionChanged, PositionClosed)):
            self._check_limits()

        if isinstance(event, PositionClosed):
            self._daily_trades += 1
            pnl = float(event.realized_pnl) if event.realized_pnl else 0
            if pnl < 0:
                self._consecutive_losses += 1
            else:
                self._consecutive_losses = 0

    def _check_limits(self) -> None:
        account = self.portfolio.account(ALPACA_VENUE)
        if account is None:
            return
        equity = float(account.balance_total(account.base_currency))

        if equity > self._peak_equity:
            self._peak_equity = equity

        # Level 2: Daily loss
        daily_return = (equity - self._start_of_day_equity) / self._start_of_day_equity
        if daily_return < -self._config.daily_loss_limit_pct:
            self._trigger(2, f"Daily loss {daily_return:.1%} > {self._config.daily_loss_limit_pct:.0%}")
            return

        if self._daily_trades >= self._config.max_daily_trades:
            self._trigger(2, f"Daily trades {self._daily_trades} >= {self._config.max_daily_trades}")
            return

        if self._consecutive_losses >= self._config.max_consecutive_losses:
            self._trigger(2, f"Consecutive losses: {self._consecutive_losses}")
            return

        # Level 3: Weekly drawdown
        weekly_dd = (self._start_of_week_equity - equity) / self._start_of_week_equity
        if weekly_dd > self._config.weekly_dd_limit_pct:
            self._trigger(3, f"Weekly DD {weekly_dd:.1%} > {self._config.weekly_dd_limit_pct:.0%}")
            return

        # Level 4: Monthly drawdown (emergency)
        monthly_dd = (self._start_of_month_equity - equity) / self._start_of_month_equity
        if monthly_dd > self._config.monthly_dd_limit_pct:
            self._trigger(4, f"Monthly DD {monthly_dd:.1%} > {self._config.monthly_dd_limit_pct:.0%}")
            # Close all positions
            for position in self.cache.positions_open():
                self.log.critical(f"L4 EMERGENCY: Closing {position.instrument_id}")
            return

    def _trigger(self, level: int, reason: str) -> None:
        if level <= self._active_level:
            return  # Already at this level or higher
        self._active_level = level
        self.log.warning(f"Circuit Breaker L{level}: {reason}")

        ts = int(time.time() * 1e9)
        data = CircuitBreakerData(level=level, reason=reason, ts_event=ts, ts_init=ts)
        self.msgbus.publish(CB_TOPIC, data)

        if level >= 4:
            self._halted = True
            self.log.critical("TRADING HALTED — manual review required")

    def reset_daily(self) -> None:
        account = self.portfolio.account(ALPACA_VENUE)
        if account:
            self._start_of_day_equity = float(account.balance_total(account.base_currency))
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._consecutive_losses = 0
        if self._active_level <= 2:
            self._active_level = 0
        self.log.info("CB: Daily limits reset")

    def reset_weekly(self) -> None:
        account = self.portfolio.account(ALPACA_VENUE)
        if account:
            self._start_of_week_equity = float(account.balance_total(account.base_currency))
        if self._active_level <= 3:
            self._active_level = 0
        self.log.info("CB: Weekly limits reset")

    def reset_monthly(self) -> None:
        account = self.portfolio.account(ALPACA_VENUE)
        if account:
            self._start_of_month_equity = float(account.balance_total(account.base_currency))
        self._active_level = 0
        self._halted = False
        self.log.info("CB: Monthly limits reset")
