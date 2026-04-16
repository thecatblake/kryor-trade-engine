"""QuestDB bar writer — NT Actor that stores price bars into QuestDB for Grafana charts.

Uses QuestDB's InfluxDB Line Protocol (ILP) for high-speed ingestion.
Grafana reads via QuestDB's PostgreSQL wire protocol (port 8812).
"""

from __future__ import annotations

import socket
import time
from datetime import datetime

from nautilus_trader.common.actor import Actor
from nautilus_trader.config import ActorConfig
from nautilus_trader.model.data import Bar

from kryor.adapters.alpaca.history import fetch_historical_bars


class QuestDBWriterConfig(ActorConfig, frozen=True):
    questdb_host: str = "localhost"
    questdb_ilp_port: int = 9009
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    symbols: list[str] = []
    preload_days: int = 400


class QuestDBWriterActor(Actor):
    """Writes price bars to QuestDB via ILP. Also preloads historical data on start."""

    def __init__(self, config: QuestDBWriterConfig) -> None:
        super().__init__(config=config)
        self._config = config
        self._sock: socket.socket | None = None

    def on_start(self) -> None:
        self.log.info("QuestDBWriter starting")

        # Connect to QuestDB ILP
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.connect((self._config.questdb_host, self._config.questdb_ilp_port))
            self.log.info(
                f"Connected to QuestDB ILP at "
                f"{self._config.questdb_host}:{self._config.questdb_ilp_port}"
            )
        except Exception as e:
            self.log.error(f"Failed to connect to QuestDB: {e}")
            self._sock = None
            return

        # Preload historical bars
        if self._config.alpaca_api_key and self._config.symbols:
            self._preload_history()

    def on_bar(self, bar: Bar) -> None:
        """Write incoming live bars to QuestDB."""
        self._write_bar(bar)

    def on_stop(self) -> None:
        if self._sock:
            self._sock.close()

    def _preload_history(self) -> None:
        """Fetch historical bars and write to QuestDB."""
        total = 0
        for sym in self._config.symbols:
            bars = fetch_historical_bars(
                self._config.alpaca_api_key,
                self._config.alpaca_secret_key,
                sym,
                days=self._config.preload_days,
            )
            for bar in bars:
                self._write_bar(bar)
            total += len(bars)

        # Flush
        if self._sock:
            try:
                self._sock.sendall(b"")
            except Exception:
                pass
        self.log.info(f"Preloaded {total} bars for {len(self._config.symbols)} symbols to QuestDB")

    def _write_bar(self, bar: Bar) -> None:
        """Write a single bar to QuestDB via ILP (InfluxDB Line Protocol)."""
        if not self._sock:
            return

        symbol = bar.bar_type.instrument_id.symbol.value
        ts_ns = bar.ts_event

        # ILP format: table,tag=value column=value timestamp_ns
        line = (
            f"bars,symbol={symbol} "
            f"open={float(bar.open)},high={float(bar.high)},"
            f"low={float(bar.low)},close={float(bar.close)},"
            f"volume={float(bar.volume)} "
            f"{ts_ns}\n"
        )

        try:
            self._sock.sendall(line.encode())
        except (BrokenPipeError, ConnectionResetError) as e:
            self.log.warning(f"QuestDB write failed, reconnecting: {e}")
            self._reconnect()
            try:
                self._sock.sendall(line.encode())
            except Exception:
                pass

    def _reconnect(self) -> None:
        try:
            if self._sock:
                self._sock.close()
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.connect((self._config.questdb_host, self._config.questdb_ilp_port))
        except Exception as e:
            self.log.error(f"QuestDB reconnect failed: {e}")
            self._sock = None
