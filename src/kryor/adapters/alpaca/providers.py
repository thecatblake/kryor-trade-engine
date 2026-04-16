"""Alpaca instrument provider — creates NT Equity instruments from Alpaca assets."""

from __future__ import annotations

import time
from decimal import Decimal

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus

from nautilus_trader.common.providers import InstrumentProvider
from nautilus_trader.model.identifiers import InstrumentId, Symbol
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Currency, Price, Quantity

from kryor.adapters.alpaca.constants import ALPACA_VENUE


class AlpacaInstrumentProvider(InstrumentProvider):
    """Provides US equity instruments from Alpaca."""

    def __init__(self, api_key: str, secret_key: str, paper: bool = True) -> None:
        super().__init__()
        self._client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
        )
        self._loaded = False

    async def load_all_async(self, filters: dict | None = None) -> None:
        if self._loaded:
            return
        await self.load_ids_async([], filters)

    async def load_ids_async(
        self,
        instrument_ids: list[InstrumentId],
        filters: dict | None = None,
    ) -> None:
        symbols = [iid.symbol.value for iid in instrument_ids] if instrument_ids else []
        assets = self._client.get_all_assets(
            GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)
        )

        ts = int(time.time() * 1e9)
        usd = Currency.from_str("USD")

        for asset in assets:
            if symbols and asset.symbol not in symbols:
                continue
            if not asset.tradable:
                continue

            instrument_id = InstrumentId(Symbol(asset.symbol), ALPACA_VENUE)
            equity = Equity(
                instrument_id=instrument_id,
                raw_symbol=Symbol(asset.symbol),
                currency=usd,
                price_precision=2,
                price_increment=Price.from_str("0.01"),
                lot_size=Quantity.from_int(1),
                ts_event=ts,
                ts_init=ts,
                maker_fee=Decimal("0"),
                taker_fee=Decimal("0"),
            )
            self.add(equity)

        self._loaded = True

    def load_symbols(self, symbols: list[str]) -> None:
        """Synchronous load for specific symbols."""
        ts = int(time.time() * 1e9)
        usd = Currency.from_str("USD")

        for sym in symbols:
            instrument_id = InstrumentId(Symbol(sym), ALPACA_VENUE)
            if instrument_id in self._instruments:
                continue
            equity = Equity(
                instrument_id=instrument_id,
                raw_symbol=Symbol(sym),
                currency=usd,
                price_precision=2,
                price_increment=Price.from_str("0.01"),
                lot_size=Quantity.from_int(1),
                ts_event=ts,
                ts_init=ts,
                maker_fee=Decimal("0"),
                taker_fee=Decimal("0"),
            )
            self.add(equity)
