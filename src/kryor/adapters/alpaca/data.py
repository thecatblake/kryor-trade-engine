"""Alpaca LiveMarketDataClient — REST polling for daily bars.

Alpaca's WebSocket only streams 1-min bars. For daily bar strategies,
we poll the REST API after market close (16:30 EST / 06:30 JST).
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from nautilus_trader.common.enums import LogColor
from nautilus_trader.live.data_client import LiveMarketDataClient
from nautilus_trader.model.data import Bar, BarSpecification, BarType
from nautilus_trader.model.enums import (
    AggregationSource,
    BarAggregation,
    PriceType,
)
from nautilus_trader.model.identifiers import InstrumentId, Symbol
from nautilus_trader.model.objects import Price, Quantity

from kryor.adapters.alpaca.config import AlpacaDataClientConfig
from kryor.adapters.alpaca.constants import ALPACA_VENUE
from kryor.adapters.alpaca.providers import AlpacaInstrumentProvider

# US market close: 16:00 EST. Poll at 16:30 EST to ensure bars are available.
POLL_DELAY_AFTER_CLOSE_SECONDS = 30 * 60  # 30 min after close
POLL_INTERVAL_SECONDS = 60 * 60  # Retry every hour if bars not found


class AlpacaDataClient(LiveMarketDataClient):
    """NautilusTrader data client — REST daily bar polling (no WebSocket)."""

    def __init__(self, loop, client_id, venue, msgbus, cache, clock,
                 instrument_provider: AlpacaInstrumentProvider,
                 config: AlpacaDataClientConfig) -> None:
        super().__init__(loop=loop, client_id=client_id, venue=venue,
                         msgbus=msgbus, cache=cache, clock=clock,
                         instrument_provider=instrument_provider, config=config)
        self._config = config
        self._hist_client = StockHistoricalDataClient(
            api_key=config.api_key, secret_key=config.secret_key)
        self._subscribed_bars: set[str] = set()
        self._poll_task: asyncio.Task | None = None
        self._last_bar_dates: dict[str, str] = {}  # symbol → last bar date "2026-04-17"

    async def _connect(self) -> None:
        self._log.info("Alpaca data client connected (REST polling mode)", LogColor.GREEN)

    async def _disconnect(self) -> None:
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
        self._log.info("Alpaca data client disconnected")

    # ── Subscriptions ──────────────────────────────────────

    async def _subscribe_bars(self, command) -> None:
        symbol = command.bar_type.instrument_id.symbol.value
        if symbol in self._subscribed_bars:
            return
        self._subscribed_bars.add(symbol)
        self._log.info(f"Subscribed bars (REST poll): {symbol}")

        # Start polling loop if not already running
        if self._poll_task is None or self._poll_task.done():
            self._poll_task = self._loop.create_task(self._poll_loop())

    async def _unsubscribe_bars(self, command) -> None:
        self._subscribed_bars.discard(command.bar_type.instrument_id.symbol.value)

    # ── REST Polling Loop ──────────────────────────────────

    async def _poll_loop(self) -> None:
        """Poll for new daily bars periodically."""
        self._log.info(
            f"Daily bar polling started for {len(self._subscribed_bars)} symbols. "
            f"Checking every {POLL_INTERVAL_SECONDS // 60} minutes."
        )

        # Initial poll immediately (catch up on missed bars)
        await self._fetch_and_publish_bars()

        while True:
            try:
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
                await self._fetch_and_publish_bars()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log.error(f"Poll error: {e}")
                await asyncio.sleep(60)  # Wait 1 min on error

    async def _fetch_and_publish_bars(self) -> None:
        """Fetch latest daily bars for all subscribed symbols and publish new ones."""
        if not self._subscribed_bars:
            return

        symbols = list(self._subscribed_bars)
        new_bars_count = 0

        for sym in symbols:
            try:
                req = StockBarsRequest(
                    symbol_or_symbols=sym,
                    timeframe=TimeFrame.Day,
                    limit=5,  # Last 5 bars to catch any missed
                )
                result = self._hist_client.get_stock_bars(req)
                bars_data = result.data.get(sym, [])

                instrument_id = InstrumentId(Symbol(sym), ALPACA_VENUE)
                bar_spec = BarSpecification(1, BarAggregation.DAY, PriceType.LAST)
                bar_type = BarType(instrument_id, bar_spec, AggregationSource.EXTERNAL)

                for ab in bars_data:
                    bar_date = ab.timestamp.strftime("%Y-%m-%d")
                    last_date = self._last_bar_dates.get(sym, "")

                    if bar_date <= last_date:
                        continue  # Already published

                    ts = int(ab.timestamp.timestamp() * 1e9)
                    bar = Bar(
                        bar_type=bar_type,
                        open=Price.from_str(f"{ab.open:.2f}"),
                        high=Price.from_str(f"{ab.high:.2f}"),
                        low=Price.from_str(f"{ab.low:.2f}"),
                        close=Price.from_str(f"{ab.close:.2f}"),
                        volume=Quantity.from_int(int(ab.volume)),
                        ts_event=ts,
                        ts_init=ts,
                    )
                    self._handle_data(bar)
                    self._last_bar_dates[sym] = bar_date
                    new_bars_count += 1

            except Exception as e:
                self._log.warning(f"Failed to fetch bars for {sym}: {e}")

        if new_bars_count > 0:
            self._log.info(f"Published {new_bars_count} new daily bars")

    # ── Request (historical) ───────────────────────────────

    async def _request_bars(self, request) -> None:
        bar_type = request.bar_type
        symbol = bar_type.instrument_id.symbol.value
        limit = request.limit or 500
        tf = self._to_alpaca_timeframe(bar_type.spec)
        if tf is None:
            self._log.warning(f"Unsupported bar spec: {bar_type.spec}")
            return
        try:
            req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=tf, limit=limit)
            alpaca_bars = self._hist_client.get_stock_bars(req)
            bars = []
            for ab in alpaca_bars.data.get(symbol, []):
                ts = int(ab.timestamp.timestamp() * 1e9)
                bars.append(Bar(
                    bar_type=bar_type,
                    open=Price.from_str(f"{ab.open:.2f}"),
                    high=Price.from_str(f"{ab.high:.2f}"),
                    low=Price.from_str(f"{ab.low:.2f}"),
                    close=Price.from_str(f"{ab.close:.2f}"),
                    volume=Quantity.from_int(int(ab.volume)),
                    ts_event=ts, ts_init=ts,
                ))
            self._handle_bars(bar_type, bars, None, request.id)
            self._log.info(f"Received {len(bars)} historical bars for {symbol}")
        except Exception as e:
            self._log.error(f"Failed to request bars for {symbol}: {e}")

    # ── Helpers ────────────────────────────────────────────

    @staticmethod
    def _to_alpaca_timeframe(spec: BarSpecification) -> TimeFrame | None:
        agg = spec.aggregation
        if agg == BarAggregation.MINUTE:
            return TimeFrame.Minute
        elif agg == BarAggregation.HOUR:
            return TimeFrame.Hour
        elif agg == BarAggregation.DAY:
            return TimeFrame.Day
        elif agg == BarAggregation.WEEK:
            return TimeFrame.Week
        elif agg == BarAggregation.MONTH:
            return TimeFrame.Month
        return None

    # ── Stubs ──────────────────────────────────────────────

    async def _subscribe_quote_ticks(self, c) -> None: pass
    async def _unsubscribe_quote_ticks(self, c) -> None: pass
    async def _subscribe_trade_ticks(self, c) -> None: pass
    async def _unsubscribe_trade_ticks(self, c) -> None: pass
    async def _subscribe_instrument(self, c) -> None:
        instrument = self.instrument_provider.find(c.instrument_id)
        if instrument:
            self._handle_data(instrument)
    async def _subscribe_instruments(self, c) -> None:
        for instrument in self.instrument_provider.get_all().values():
            self._handle_data(instrument)
    async def _unsubscribe_instrument(self, c) -> None: pass
    async def _unsubscribe_instruments(self, c) -> None: pass
    async def _subscribe_order_book_deltas(self, c) -> None: pass
    async def _subscribe_order_book_depth(self, c) -> None: pass
    async def _unsubscribe_order_book_deltas(self, c) -> None: pass
    async def _unsubscribe_order_book_depth(self, c) -> None: pass
    async def _subscribe_instrument_status(self, c) -> None: pass
    async def _unsubscribe_instrument_status(self, c) -> None: pass
    async def _subscribe_instrument_close(self, c) -> None: pass
    async def _unsubscribe_instrument_close(self, c) -> None: pass
    async def _subscribe_mark_prices(self, c) -> None: pass
    async def _unsubscribe_mark_prices(self, c) -> None: pass
    async def _subscribe_index_prices(self, c) -> None: pass
    async def _unsubscribe_index_prices(self, c) -> None: pass
    async def _subscribe_funding_rates(self, c) -> None: pass
    async def _unsubscribe_funding_rates(self, c) -> None: pass
    async def _subscribe_option_greeks(self, c) -> None: pass
    async def _unsubscribe_option_greeks(self, c) -> None: pass
    async def _subscribe(self, c) -> None: pass
    async def _unsubscribe(self, c) -> None: pass
    async def _request(self, r) -> None: pass
    async def _request_instrument(self, r) -> None:
        instrument = self.instrument_provider.find(r.instrument_id)
        if instrument:
            self._handle_data(instrument)
    async def _request_instruments(self, r) -> None:
        for instrument in self.instrument_provider.get_all().values():
            self._handle_data(instrument)
    async def _request_quote_ticks(self, r) -> None: pass
    async def _request_trade_ticks(self, r) -> None: pass
    async def _request_order_book_snapshot(self, r) -> None: pass
    async def _request_order_book_deltas(self, r) -> None: pass
    async def _request_order_book_depth(self, r) -> None: pass
