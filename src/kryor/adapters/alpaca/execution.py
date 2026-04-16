"""Alpaca LiveExecutionClient — order execution via alpaca-py."""

from __future__ import annotations

import time
from decimal import Decimal

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce, OrderClass, QueryOrderStatus
from alpaca.trading.requests import LimitOrderRequest, GetOrdersRequest

from nautilus_trader.common.enums import LogColor
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.execution.messages import (
    CancelAllOrders,
    CancelOrder,
    ModifyOrder,
    SubmitOrder,
)
from nautilus_trader.live.execution_client import LiveExecutionClient
from nautilus_trader.model.enums import (
    AccountType,
    LiquiditySide,
    OmsType,
    OrderSide as NTOrderSide,
    OrderType as NTOrderType,
)
from nautilus_trader.model.identifiers import (
    AccountId,
    ClientOrderId,
    InstrumentId,
    Symbol,
    TradeId,
    VenueOrderId,
)
from nautilus_trader.model.objects import Currency, Money, Price, Quantity

from kryor.adapters.alpaca.config import AlpacaExecClientConfig
from kryor.adapters.alpaca.constants import ALPACA_VENUE
from kryor.adapters.alpaca.providers import AlpacaInstrumentProvider


class AlpacaExecutionClient(LiveExecutionClient):
    """NautilusTrader execution client for Alpaca Markets."""

    def __init__(
        self,
        loop,
        client_id,
        venue,
        msgbus,
        cache,
        clock,
        instrument_provider: AlpacaInstrumentProvider,
        config: AlpacaExecClientConfig,
    ) -> None:
        super().__init__(
            loop=loop,
            client_id=client_id,
            venue=venue,
            oms_type=OmsType.NETTING,
            account_type=AccountType.CASH,
            base_currency=Currency.from_str("USD"),
            instrument_provider=instrument_provider,
            msgbus=msgbus,
            cache=cache,
            clock=clock,
            config=config,
        )
        self._config = config
        self._client = TradingClient(
            api_key=config.api_key,
            secret_key=config.secret_key,
            paper=config.paper,
        )

    async def _connect(self) -> None:
        from nautilus_trader.model.objects import AccountBalance

        account = self._client.get_account()
        account_id = AccountId(f"ALPACA-{'paper' if self._config.paper else 'live'}")
        self._set_account_id(account_id)

        usd = Currency.from_str("USD")
        equity = Decimal(str(account.equity))
        cash = Decimal(str(account.cash))

        self.generate_account_state(
            balances=[
                AccountBalance(
                    total=Money(equity, usd),
                    locked=Money(equity - cash, usd),
                    free=Money(cash, usd),
                ),
            ],
            margins=[],
            reported=True,
            ts_event=int(time.time() * 1e9),
        )
        self._log.info(
            f"Alpaca exec connected: equity=${float(account.equity):,.2f}, "
            f"cash=${float(account.cash):,.2f}",
            LogColor.GREEN,
        )

    async def _disconnect(self) -> None:
        self._log.info("Alpaca exec client disconnected")

    # ── Order Submission ───────────────────────────────────

    def _submit_order(self, command: SubmitOrder) -> None:
        self._loop.create_task(self._submit_order_async(command))

    async def _submit_order_async(self, command: SubmitOrder) -> None:
        order = command.order
        symbol = order.instrument_id.symbol.value

        try:
            alpaca_side = OrderSide.BUY if order.side == NTOrderSide.BUY else OrderSide.SELL

            if order.order_type == NTOrderType.LIMIT:
                limit_price = float(order.price)

                # Check for bracket order (stop_loss and take_profit attached)
                if order.linked_order_ids:
                    # Bracket handled via OrderList in NT
                    req = LimitOrderRequest(
                        symbol=symbol,
                        qty=float(order.quantity),
                        side=alpaca_side,
                        type=OrderType.LIMIT,
                        time_in_force=TimeInForce.DAY,
                        limit_price=round(limit_price, 2),
                    )
                else:
                    req = LimitOrderRequest(
                        symbol=symbol,
                        qty=float(order.quantity),
                        side=alpaca_side,
                        type=OrderType.LIMIT,
                        time_in_force=TimeInForce.DAY,
                        limit_price=round(limit_price, 2),
                    )
            else:
                # Market order fallback
                from alpaca.trading.requests import MarketOrderRequest
                req = MarketOrderRequest(
                    symbol=symbol,
                    qty=float(order.quantity),
                    side=alpaca_side,
                    type=OrderType.MARKET,
                    time_in_force=TimeInForce.DAY,
                )

            result = self._client.submit_order(req)
            venue_order_id = VenueOrderId(str(result.id))
            ts = int(time.time() * 1e9)

            self.generate_order_submitted(
                strategy_id=order.strategy_id,
                instrument_id=order.instrument_id,
                client_order_id=order.client_order_id,
                ts_event=ts,
            )

            self.generate_order_accepted(
                strategy_id=order.strategy_id,
                instrument_id=order.instrument_id,
                client_order_id=order.client_order_id,
                venue_order_id=venue_order_id,
                ts_event=ts,
            )

            # Check if already filled
            if result.filled_qty and float(result.filled_qty) > 0:
                self.generate_order_filled(
                    strategy_id=order.strategy_id,
                    instrument_id=order.instrument_id,
                    client_order_id=order.client_order_id,
                    venue_order_id=venue_order_id,
                    trade_id=TradeId(str(result.id)),
                    venue_position_id=None,
                    order_side=order.side,
                    order_type=order.order_type,
                    last_qty=Quantity.from_str(str(result.filled_qty)),
                    last_px=Price.from_str(str(result.filled_avg_price)),
                    quote_currency=Currency.from_str("USD"),
                    commission=Money(Decimal("0"), Currency.from_str("USD")),
                    liquidity_side=LiquiditySide.TAKER,
                    ts_event=ts,
                )

            self._log.info(
                f"Order submitted: {order.side} {order.quantity} {symbol} "
                f"@ {order.price} → {result.status} (id={result.id})"
            )

        except Exception as e:
            ts = int(time.time() * 1e9)
            self.generate_order_rejected(
                strategy_id=order.strategy_id,
                instrument_id=order.instrument_id,
                client_order_id=order.client_order_id,
                reason=str(e),
                ts_event=ts,
            )
            self._log.error(f"Order rejected: {e}")

    # ── Order Cancellation ─────────────────────────────────

    def _cancel_order(self, command: CancelOrder) -> None:
        self._loop.create_task(self._cancel_order_async(command))

    async def _cancel_order_async(self, command: CancelOrder) -> None:
        try:
            venue_order_id = command.venue_order_id
            if venue_order_id:
                self._client.cancel_order_by_id(venue_order_id.value)
                ts = int(time.time() * 1e9)
                self.generate_order_canceled(
                    strategy_id=command.strategy_id,
                    instrument_id=command.instrument_id,
                    client_order_id=command.client_order_id,
                    venue_order_id=venue_order_id,
                    ts_event=ts,
                )
                self._log.info(f"Order cancelled: {venue_order_id}")
        except Exception as e:
            self._log.error(f"Cancel failed: {e}")

    def _cancel_all_orders(self, command: CancelAllOrders) -> None:
        self._loop.create_task(self._cancel_all_async(command))

    async def _cancel_all_async(self, command: CancelAllOrders) -> None:
        try:
            self._client.cancel_orders()
            self._log.warning("All orders cancelled")
        except Exception as e:
            self._log.error(f"Cancel all failed: {e}")

    def _modify_order(self, command: ModifyOrder) -> None:
        # Alpaca doesn't support order modification — cancel and re-submit
        self._log.warning("Order modify not supported by Alpaca, cancel and re-submit")

    # ── Reconciliation ─────────────────────────────────────

    async def generate_order_status_reports(
        self,
        instrument_id: InstrumentId | None = None,
        start=None,
        end=None,
        open_only: bool = False,
    ) -> list:
        return []

    async def generate_fill_reports(
        self,
        instrument_id: InstrumentId | None = None,
        venue_order_id=None,
        start=None,
        end=None,
    ) -> list:
        return []

    async def generate_position_status_reports(
        self,
        instrument_id: InstrumentId | None = None,
        start=None,
        end=None,
    ) -> list:
        return []

    # ── Stubs ──────────────────────────────────────────────

    def _batch_cancel_orders(self, command) -> None:
        self._cancel_all_orders(command)

    def _query_order(self, command) -> None:
        pass

    def _submit_order_list(self, command) -> None:
        # Submit each order individually
        for order in command.order_list.orders:
            submit_cmd = SubmitOrder(
                trader_id=command.trader_id,
                strategy_id=command.strategy_id,
                order=order,
                command_id=UUID4(),
                ts_init=int(time.time() * 1e9),
                position_id=command.position_id,
            )
            self._submit_order(submit_cmd)
