from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class Regime(str, Enum):
    BULL = "bull"
    NEUTRAL = "neutral"
    BEAR = "bear"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass(frozen=True)
class NormalizedBar:
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float = 0.0
    trade_count: int = 0
    timeframe: str = "1d"


@dataclass(frozen=True)
class Signal:
    timestamp: datetime
    symbol: str
    direction: Side | None  # None = hold
    score: float  # -1.0 to +1.0
    strategy: str
    stop_loss: float | None = None
    take_profit: float | None = None
    max_hold_days: int | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RegimeState:
    timestamp: datetime
    regime: Regime
    probability: float
    bull_prob: float
    neutral_prob: float
    bear_prob: float
    duration_days: int = 0


@dataclass
class PortfolioState:
    equity: float
    cash: float
    daily_pnl: float = 0.0
    weekly_drawdown: float = 0.0
    monthly_drawdown: float = 0.0
    daily_trade_count: int = 0
    consecutive_losses: int = 0
    orders_last_minute: int = 0
    peak_equity: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)

    def update_drawdowns(self) -> None:
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        if self.peak_equity > 0:
            dd = (self.peak_equity - self.equity) / self.peak_equity
            self.monthly_drawdown = max(self.monthly_drawdown, dd)
            self.weekly_drawdown = max(self.weekly_drawdown, dd)


@dataclass
class Position:
    symbol: str
    side: Side
    qty: int
    avg_entry_price: float
    current_price: float = 0.0
    stop_loss: float | None = None
    take_profit: float | None = None
    opened_at: datetime | None = None
    strategy: str = ""
    max_hold_days: int | None = None

    @property
    def unrealized_pnl(self) -> float:
        mult = 1 if self.side == Side.BUY else -1
        return mult * (self.current_price - self.avg_entry_price) * self.qty

    @property
    def market_value(self) -> float:
        return self.current_price * self.qty


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    side: Side
    qty: int
    limit_price: float
    stop_loss: float | None = None
    take_profit: float | None = None
    strategy: str = ""
    signal_score: float = 0.0
    regime: Regime = Regime.NEUTRAL


@dataclass
class OrderResult:
    order_id: str
    symbol: str
    side: Side
    qty: int
    status: OrderStatus
    filled_price: float | None = None
    filled_qty: int = 0
    message: str = ""
    timestamp: datetime | None = None
