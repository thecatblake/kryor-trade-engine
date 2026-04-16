from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # Alpaca
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_paper: bool = True

    # Database
    postgres_url: str = "postgresql://kryor:kryor_dev@localhost:5432/kryor_trade"
    questdb_host: str = "localhost"
    questdb_http_port: int = 9000
    questdb_pg_port: int = 8812
    questdb_ilp_port: int = 9009
    redis_url: str = "redis://localhost:6379"

    # Trading
    initial_capital: float = 10_000.0
    max_positions: int = 10
    max_position_pct: float = 0.15
    max_sector_pct: float = 0.30
    per_trade_risk_pct: float = 0.01
    daily_loss_limit_pct: float = 0.03
    weekly_dd_limit_pct: float = 0.08
    monthly_dd_limit_pct: float = 0.15

    # Monitoring
    discord_webhook_url: str = ""
    prometheus_port: int = 8000

    # Strategy
    momentum_lookback_months: int = 12
    momentum_skip_months: int = 1
    momentum_top_pct: float = 0.10
    reversion_rsi_threshold: int = 30
    reversion_volume_mult: float = 1.5
    reversion_max_hold_days: int = 5

    @property
    def alpaca_base_url(self) -> str:
        if self.alpaca_paper:
            return "https://paper-api.alpaca.markets"
        return "https://api.alpaca.markets"
