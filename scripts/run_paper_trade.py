#!/usr/bin/env python3
"""Run the Kryor Trade Engine via NautilusTrader TradingNode (paper trading).

Usage:
    python scripts/run_paper_trade.py

Requires:
    - .env file with ALPACA_API_KEY and ALPACA_SECRET_KEY
    - Docker services running (docker compose up -d)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

from nautilus_trader.config import (
    LoggingConfig,
    TradingNodeConfig,
)
from nautilus_trader.live.node import TradingNode

from kryor.adapters.alpaca import (
    AlpacaDataClientConfig,
    AlpacaExecClientConfig,
    AlpacaLiveDataClientFactory,
    AlpacaLiveExecClientFactory,
)
from kryor.adapters.alpaca.providers import AlpacaInstrumentProvider
from kryor.api.control import start_api_server
from kryor.data.questdb_writer import QuestDBWriterActor, QuestDBWriterConfig
from kryor.monitoring.metrics_actor import MetricsActor, MetricsActorConfig
from kryor.regime.hmm import RegimeActor, RegimeActorConfig
from kryor.risk.circuit_breaker import CircuitBreakerActor, CircuitBreakerConfig
from kryor.strategy.momentum import MomentumConfig, MomentumStrategy
from kryor.strategy.mean_reversion import MeanReversionConfig, MeanReversionStrategy
from kryor.strategy.ml_signal import MLSignalConfig, MLSignalStrategy

# ── Universe (determined by capital at startup) ──────────────

from kryor.core.portfolio_config import PortfolioConfig


def main() -> None:
    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    paper = os.environ.get("ALPACA_PAPER", "true").lower() == "true"

    if not api_key or not secret_key:
        print("ERROR: Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")
        print("  Get free keys at: https://app.alpaca.markets/")
        sys.exit(1)

    # ── Pre-load instruments ──────────────────────────────

    # Determine universe from capital
    initial_capital = float(os.environ.get("INITIAL_CAPITAL", "2000"))
    pf_config = PortfolioConfig.from_equity(initial_capital)
    UNIVERSE = pf_config.get_universe()
    print(f"Capital: ${initial_capital:,.0f} → Universe: {len(UNIVERSE)} symbols")
    print(pf_config.describe())

    provider = AlpacaInstrumentProvider(api_key, secret_key, paper)
    provider.load_symbols(UNIVERSE)
    print(f"Loaded {len(UNIVERSE)} instruments")

    # ── TradingNode Configuration ─────────────────────────

    config = TradingNodeConfig(
        trader_id="KRYOR-001",
        logging=LoggingConfig(log_level="INFO"),
        data_clients={
            "ALPACA": AlpacaDataClientConfig(
                api_key=api_key,
                secret_key=secret_key,
                paper=paper,
            ),
        },
        exec_clients={
            "ALPACA": AlpacaExecClientConfig(
                api_key=api_key,
                secret_key=secret_key,
                paper=paper,
            ),
        },
    )

    # ── Start Prometheus metrics server (must be in main thread before node starts) ──
    from prometheus_client import start_http_server as start_prom
    try:
        start_prom(8000)
        print("Prometheus metrics server on :8000")
    except OSError:
        print("Prometheus port 8000 already in use")

    # Import metrics to register them in the default registry
    import kryor.monitoring.metrics_actor  # noqa: F401

    # ── Start Trade Control API (:8001) ───────────────────
    from alpaca.trading.client import TradingClient
    trading_client = TradingClient(api_key=api_key, secret_key=secret_key, paper=paper)
    start_api_server(trading_client, port=8001)
    print("Trade Control API on :8001 (docs: http://localhost:8001/docs)")

    # ── Build Node ────────────────────────────────────────

    node = TradingNode(config)
    node.add_data_client_factory("ALPACA", AlpacaLiveDataClientFactory)
    node.add_exec_client_factory("ALPACA", AlpacaLiveExecClientFactory)

    # ── Strategies (pluggable — swap or add strategies here) ──

    momentum = MomentumStrategy(
        config=MomentumConfig(
            strategy_id="MOM-001",
            symbols=UNIVERSE,
            alpaca_api_key=api_key,
            alpaca_secret_key=secret_key,
            lookback_days=252,
            skip_days=21,
        ),
    )

    mean_rev = MeanReversionStrategy(
        config=MeanReversionConfig(
            strategy_id="MR-001",
            symbols=UNIVERSE,
            alpaca_api_key=api_key,
            alpaca_secret_key=secret_key,
            rsi_threshold=30,
            volume_mult=1.5,
            max_hold_days=5,
        ),
    )

    ml_signal = MLSignalStrategy(
        config=MLSignalConfig(
            strategy_id="ML-001",
            symbols=UNIVERSE,
            alpaca_api_key=api_key,
            alpaca_secret_key=secret_key,
            model_path="models/lgbm_signal_v1.pkl",
            buy_threshold=0.45,
            max_hold_days=10,
        ),
    )

    node.trader.add_strategy(momentum)
    node.trader.add_strategy(mean_rev)
    node.trader.add_strategy(ml_signal)

    # ── Actors (order matters: metrics first so it subscribes before regime publishes) ──

    metrics_actor = MetricsActor(
        config=MetricsActorConfig(
            component_id="METRICS-001",
            prometheus_port=8000,
            update_interval_secs=10.0,
        ),
    )
    node.trader.add_actor(metrics_actor)

    cb_actor = CircuitBreakerActor(
        config=CircuitBreakerConfig(component_id="CB-001"),
    )
    node.trader.add_actor(cb_actor)

    regime_actor = RegimeActor(
        config=RegimeActorConfig(component_id="REGIME-001"),
    )
    node.trader.add_actor(regime_actor)

    questdb_writer = QuestDBWriterActor(
        config=QuestDBWriterConfig(
            component_id="QUESTDB-001",
            alpaca_api_key=api_key,
            alpaca_secret_key=secret_key,
            symbols=UNIVERSE,
            preload_days=400,
        ),
    )
    node.trader.add_actor(questdb_writer)

    # ── Run ───────────────────────────────────────────────

    node.build()
    node.run()


if __name__ == "__main__":
    main()
