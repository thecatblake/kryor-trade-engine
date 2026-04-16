"""Factories for creating Alpaca data and execution clients."""

from __future__ import annotations

import asyncio

from nautilus_trader.cache.cache import Cache
from nautilus_trader.common.component import LiveClock, MessageBus
from nautilus_trader.live.factories import LiveDataClientFactory, LiveExecClientFactory
from nautilus_trader.model.identifiers import ClientId

from kryor.adapters.alpaca.config import AlpacaDataClientConfig, AlpacaExecClientConfig
from kryor.adapters.alpaca.constants import ALPACA_VENUE
from kryor.adapters.alpaca.data import AlpacaDataClient
from kryor.adapters.alpaca.execution import AlpacaExecutionClient
from kryor.adapters.alpaca.providers import AlpacaInstrumentProvider


class AlpacaLiveDataClientFactory(LiveDataClientFactory):
    @staticmethod
    def create(
        loop: asyncio.AbstractEventLoop,
        name: str,
        config: AlpacaDataClientConfig,
        msgbus: MessageBus,
        cache: Cache,
        clock: LiveClock,
    ) -> AlpacaDataClient:
        provider = AlpacaInstrumentProvider(
            api_key=config.api_key,
            secret_key=config.secret_key,
            paper=config.paper,
        )
        return AlpacaDataClient(
            loop=loop,
            client_id=ClientId(name),
            venue=ALPACA_VENUE,
            msgbus=msgbus,
            cache=cache,
            clock=clock,
            instrument_provider=provider,
            config=config,
        )


class AlpacaLiveExecClientFactory(LiveExecClientFactory):
    @staticmethod
    def create(
        loop: asyncio.AbstractEventLoop,
        name: str,
        config: AlpacaExecClientConfig,
        msgbus: MessageBus,
        cache: Cache,
        clock: LiveClock,
    ) -> AlpacaExecutionClient:
        provider = AlpacaInstrumentProvider(
            api_key=config.api_key,
            secret_key=config.secret_key,
            paper=config.paper,
        )
        return AlpacaExecutionClient(
            loop=loop,
            client_id=ClientId(name),
            venue=ALPACA_VENUE,
            msgbus=msgbus,
            cache=cache,
            clock=clock,
            instrument_provider=provider,
            config=config,
        )
