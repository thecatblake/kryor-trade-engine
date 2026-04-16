from __future__ import annotations

import sys
from pathlib import Path

from discord_webhook import DiscordWebhook
from loguru import logger
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from kryor.core.settings import Settings

# ── Prometheus Metrics ──────────────────────────────────────────

# P&L
trade_pnl_total = Gauge("trade_pnl_total", "Cumulative P&L")
trade_pnl_daily = Gauge("trade_pnl_daily", "Daily P&L")
equity_current = Gauge("equity_current", "Current equity")
drawdown_current = Gauge("drawdown_current", "Current drawdown ratio")
sharpe_30d = Gauge("sharpe_ratio_30d", "30-day rolling Sharpe ratio")

# Positions
position_count = Gauge("position_count", "Open position count")
total_exposure = Gauge("total_exposure", "Total dollar exposure")

# Execution
order_count = Counter("order_count_total", "Total orders submitted", ["side", "status"])
avg_slippage = Histogram("slippage_bps", "Slippage in basis points", buckets=[1, 2, 5, 10, 20, 50])
order_latency = Histogram("order_latency_ms", "Order latency ms", buckets=[50, 100, 200, 500, 1000])

# System
regime_current = Gauge("regime_current", "Current regime: 0=bull, 1=neutral, 2=bear")
circuit_breaker_active = Gauge("circuit_breaker_active", "CB level active (0=none)")
signal_count_daily = Counter("signal_count_daily", "Signals generated today")


def setup_logging(log_dir: str = "logs") -> None:
    Path(log_dir).mkdir(exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")
    logger.add(
        f"{log_dir}/kryor_{{time:YYYY-MM-DD}}.log",
        rotation="1 day",
        retention="30 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | {name}:{function}:{line} | {message}",
    )


def start_metrics_server(port: int = 8000) -> None:
    start_http_server(port)
    logger.info(f"Prometheus metrics server on :{port}")


class DiscordNotifier:
    def __init__(self, webhook_url: str) -> None:
        self._url = webhook_url

    def send(self, message: str, level: str = "INFO") -> None:
        if not self._url:
            logger.debug(f"Discord (no webhook): {message}")
            return
        prefix = {"CRITICAL": ":rotating_light:", "HIGH": ":warning:", "MEDIUM": ":information_source:"}.get(
            level, ":white_small_square:"
        )
        try:
            webhook = DiscordWebhook(url=self._url, content=f"{prefix} **{level}** | {message}")
            webhook.execute()
        except Exception as e:
            logger.error(f"Discord send failed: {e}")


def create_notifier(settings: Settings) -> DiscordNotifier:
    return DiscordNotifier(settings.discord_webhook_url)
