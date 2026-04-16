"""Database storage layer — PostgreSQL via SQLAlchemy, Redis for cache."""

from __future__ import annotations

import json
from datetime import datetime

import redis
from loguru import logger
from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Integer,
    Numeric,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from kryor.core.models import OrderResult, OrderStatus, Position, Side
from kryor.core.settings import Settings


# ── SQLAlchemy Models ───────────────────────────────────────────


class Base(DeclarativeBase):
    pass


class OrderLog(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, server_default=func.now())
    symbol = Column(String(20), nullable=False)
    side = Column(String(4), nullable=False)
    order_type = Column(String(10), nullable=False, default="limit")
    qty = Column(Integer, nullable=False)
    limit_price = Column(Numeric(12, 4))
    filled_price = Column(Numeric(12, 4))
    filled_qty = Column(Integer, default=0)
    status = Column(String(20), nullable=False)
    broker = Column(String(10), nullable=False, default="alpaca")
    strategy = Column(String(30))
    signal_score = Column(Numeric(5, 4))
    regime_state = Column(String(10))
    slippage_bps = Column(Numeric(8, 4))


class DailyPerformance(Base):
    __tablename__ = "daily_performance"

    date = Column(Date, primary_key=True)
    total_equity = Column(Numeric(14, 2))
    daily_pnl = Column(Numeric(12, 2))
    daily_return = Column(Numeric(8, 6))
    cumulative_return = Column(Numeric(8, 6))
    drawdown = Column(Numeric(8, 6))
    sharpe_rolling_30d = Column(Numeric(6, 4))
    regime_state = Column(String(10))
    trade_count = Column(Integer)
    win_count = Column(Integer)
    loss_count = Column(Integer)


class StrategyParam(Base):
    __tablename__ = "strategy_params"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy = Column(String(30), nullable=False)
    version = Column(Integer, nullable=False)
    params = Column(JSONB, nullable=False)
    active = Column(Boolean, default=False)
    created_at = Column(DateTime, server_default=func.now())
    notes = Column(Text)


# ── Store Classes ───────────────────────────────────────────────


class PostgresStore:
    def __init__(self, settings: Settings) -> None:
        self.engine = create_engine(settings.postgres_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def create_tables(self) -> None:
        Base.metadata.create_all(self.engine)
        logger.info("PostgreSQL tables created")

    def log_order(self, order_result: OrderResult, **extra: object) -> None:
        with self.SessionLocal() as session:
            log = OrderLog(
                symbol=order_result.symbol,
                side=order_result.side.value,
                qty=order_result.qty,
                status=order_result.status.value,
                filled_price=order_result.filled_price,
                filled_qty=order_result.filled_qty,
                **extra,
            )
            session.add(log)
            session.commit()

    def log_daily_performance(
        self,
        date: datetime,
        equity: float,
        daily_pnl: float,
        daily_return: float,
        cumulative_return: float,
        drawdown: float,
        regime: str,
        trades: int,
        wins: int,
        losses: int,
    ) -> None:
        with self.SessionLocal() as session:
            perf = DailyPerformance(
                date=date.date() if isinstance(date, datetime) else date,
                total_equity=equity,
                daily_pnl=daily_pnl,
                daily_return=daily_return,
                cumulative_return=cumulative_return,
                drawdown=drawdown,
                regime_state=regime,
                trade_count=trades,
                win_count=wins,
                loss_count=losses,
            )
            session.merge(perf)
            session.commit()

    def get_session(self) -> Session:
        return self.SessionLocal()


class RedisStore:
    def __init__(self, settings: Settings) -> None:
        self.client = redis.from_url(settings.redis_url, decode_responses=True)

    def set_price(self, symbol: str, price: float, bid: float = 0, ask: float = 0) -> None:
        self.client.set(
            f"price:{symbol}",
            json.dumps({"price": price, "bid": bid, "ask": ask, "ts": datetime.utcnow().isoformat()}),
        )

    def get_price(self, symbol: str) -> dict | None:
        raw = self.client.get(f"price:{symbol}")
        return json.loads(raw) if raw else None

    def set_regime(self, state: str, probability: float) -> None:
        self.client.set(
            "regime:current",
            json.dumps({"state": state, "probability": probability, "ts": datetime.utcnow().isoformat()}),
        )

    def get_regime(self) -> dict | None:
        raw = self.client.get("regime:current")
        return json.loads(raw) if raw else None

    def set_signal(self, symbol: str, direction: str, score: float, strategy: str) -> None:
        self.client.set(
            f"signal:{symbol}",
            json.dumps({
                "direction": direction,
                "score": score,
                "strategy": strategy,
                "ts": datetime.utcnow().isoformat(),
            }),
        )

    def set_portfolio(self, state: dict) -> None:
        self.client.set("portfolio:state", json.dumps(state))

    def publish(self, channel: str, data: dict) -> None:
        self.client.publish(channel, json.dumps(data))
