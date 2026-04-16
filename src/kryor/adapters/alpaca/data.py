"""Alpaca LiveMarketDataClient — feeds market data into NautilusTrader."""

from __future__ import annotations

import asyncio
import time
from decimal import Decimal

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from nautilus_trader.common.enums import LogColor
from nautilus_trader.live.data_client import LiveMarketDataClient
from nautilus_trader.model.data import Bar, BarSpecification, BarType, QuoteTick
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


class AlpacaDataClient(LiveMarketDataClient):
    """NautilusTrader data client for Alpaca Markets.

    Uses a SINGLE WebSocket connection for all subscriptions (Alpaca free tier limit).
    Subscriptions are batched and the stream is started once via a delayed task.
    """

    def __init__(self, loop, client_id, venue, msgbus, cache, clock,
                 instrument_provider: AlpacaInstrumentProvider,
                 config: AlpacaDataClientConfig) -> None:
        super().__init__(loop=loop, client_id=client_id, venue=venue,
                         msgbus=msgbus, cache=cache, clock=clock,
                         instrument_provider=instrument_provider, config=config)
        self._config = config
        self._hist_client = StockHistoricalDataClient(
            api_key=config.api_key, secret_key=config.secret_key)
        self._stream: StockDataStream | None = None
        self._stream_task: asyncio.Task | None = None
        self._stream_started = False
        self._start_stream_handle: asyncio.TimerHandle | None = None
        self._subscribed_bars: set[str] = set()
        self._subscribed_quotes: set[str] = set()

    async def _connect(self) -> None:
        self._stream = StockDataStream(
            api_key=self._config.api_key, secret_key=self._config.secret_key)
        self._log.info("Alpaca data client connected", LogColor.GREEN)

    async def _disconnect(self) -> None:
        if self._start_stream_handle:
            self._start_stream_handle.cancel()
        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
        if self._stream:
            try:
                await self._stream.close()
            except Exception:
                pass
        self._stream_started = False
        self._log.info("Alpaca data client disconnected")

    # ── Subscriptions ──────────────────────────────────────
    # Subscribe adds symbols to the stream but does NOT start it immediately.
    # A delayed task starts the stream 2s after the last subscription,
    # ensuring all symbols are batched into a single WebSocket connection.

    async def _subscribe_bars(self, command) -> None:
        symbol = command.bar_type.instrument_id.symbol.value
        if symbol in self._subscribed_bars:
            return
        self._subscribed_bars.add(symbol)
        if self._stream and not self._stream_started:
            self._stream.subscribe_bars(self._on_alpaca_bar, symbol)
        self._schedule_stream_start()
        self._log.info(f"Subscribed bars: {symbol}")

    async def _unsubscribe_bars(self, command) -> None:
        self._subscribed_bars.discard(command.bar_type.instrument_id.symbol.value)

    async def _subscribe_quote_ticks(self, command) -> None:
        symbol = command.instrument_id.symbol.value
        if symbol in self._subscribed_quotes:
            return
        self._subscribed_quotes.add(symbol)
        if self._stream and not self._stream_started:
            self._stream.subscribe_quotes(self._on_alpaca_quote, symbol)
        self._schedule_stream_start()
        self._log.info(f"Subscribed quotes: {symbol}")

    async def _unsubscribe_quote_ticks(self, command) -> None:
        self._subscribed_quotes.discard(command.instrument_id.symbol.value)

    async def _subscribe_trade_ticks(self, command) -> None:
        pass  # Not used for daily strategies

    async def _unsubscribe_trade_ticks(self, command) -> None:
        pass

    async def _subscribe_instrument(self, command) -> None:
        instrument = self.instrument_provider.find(command.instrument_id)
        if instrument:
            self._handle_data(instrument)

    async def _subscribe_instruments(self, command) -> None:
        for instrument in self.instrument_provider.get_all().values():
            self._handle_data(instrument)

    async def _unsubscribe_instrument(self, command) -> None:
        pass

    async def _unsubscribe_instruments(self, command) -> None:
        pass

    # ── Requests ───────────────────────────────────────────

    async def _request_instrument(self, request) -> None:
        instrument = self.instrument_provider.find(request.instrument_id)
        if instrument:
            self._handle_data(instrument)

    async def _request_instruments(self, request) -> None:
        for instrument in self.instrument_provider.get_all().values():
            self._handle_data(instrument)

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

    # ── Stream Management ──────────────────────────────────

    def _schedule_stream_start(self) -> None:
        """Schedule stream start 2s from now, debouncing repeated calls."""
        if self._stream_started:
            return
        if self._start_stream_handle:
            self._start_stream_handle.cancel()
        self._start_stream_handle = self._loop.call_later(
            2.0, lambda: self._loop.create_task(self._start_stream())
        )

    async def _start_stream(self) -> None:
        """Start the single WebSocket stream with all subscribed symbols."""
        if self._stream_started or not self._stream:
            return
        self._stream_started = True
        n_bars = len(self._subscribed_bars)
        n_quotes = len(self._subscribed_quotes)
        self._log.info(
            f"Starting Alpaca WebSocket stream ({n_bars} bar subs, {n_quotes} quote subs)"
        )
        self._stream_task = self._loop.create_task(self._run_stream())

    async def _run_stream(self) -> None:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                await asyncio.to_thread(self._stream.run)
                break  # Clean exit
            except asyncio.CancelledError:
                break
            except ValueError as e:
                if "connection limit" in str(e).lower():
                    self._log.warning(
                        f"Alpaca connection limit exceeded (attempt {attempt + 1}/{max_retries}). "
                        f"Waiting 30s before retry..."
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(30)
                    else:
                        self._log.error("Alpaca connection limit: max retries exhausted. Stream disabled.")
                        self._stream_started = False
                else:
                    self._log.error(f"Alpaca stream error: {e}")
                    self._stream_started = False
                    break
            except Exception as e:
                self._log.error(f"Alpaca stream error: {e}")
                self._stream_started = False
                break

    # ── Streaming Handlers ─────────────────────────────────

    async def _on_alpaca_bar(self, alpaca_bar) -> None:
        symbol = alpaca_bar.symbol
        instrument_id = InstrumentId(Symbol(symbol), ALPACA_VENUE)
        bar_spec = BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST)
        bar_type = BarType(instrument_id, bar_spec, AggregationSource.EXTERNAL)
        ts = int(alpaca_bar.timestamp.timestamp() * 1e9)
        bar = Bar(bar_type=bar_type,
                  open=Price.from_str(f"{alpaca_bar.open:.2f}"),
                  high=Price.from_str(f"{alpaca_bar.high:.2f}"),
                  low=Price.from_str(f"{alpaca_bar.low:.2f}"),
                  close=Price.from_str(f"{alpaca_bar.close:.2f}"),
                  volume=Quantity.from_int(int(alpaca_bar.volume)),
                  ts_event=ts, ts_init=ts)
        self._handle_data(bar)

    async def _on_alpaca_quote(self, alpaca_quote) -> None:
        symbol = alpaca_quote.symbol
        instrument_id = InstrumentId(Symbol(symbol), ALPACA_VENUE)
        ts = int(alpaca_quote.timestamp.timestamp() * 1e9)
        tick = QuoteTick(instrument_id=instrument_id,
                         bid_price=Price.from_str(f"{alpaca_quote.bid_price:.2f}"),
                         ask_price=Price.from_str(f"{alpaca_quote.ask_price:.2f}"),
                         bid_size=Quantity.from_int(int(alpaca_quote.bid_size)),
                         ask_size=Quantity.from_int(int(alpaca_quote.ask_size)),
                         ts_event=ts, ts_init=ts)
        self._handle_data(tick)

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
    async def _request_quote_ticks(self, r) -> None: pass
    async def _request_trade_ticks(self, r) -> None: pass
    async def _request_order_book_snapshot(self, r) -> None: pass
    async def _request_order_book_deltas(self, r) -> None: pass
    async def _request_order_book_depth(self, r) -> None: pass
