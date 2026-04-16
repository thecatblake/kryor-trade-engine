"""Trade control API — FastAPI server running inside the TradingNode process.

Provides REST endpoints for:
  - Manual order submission (buy/sell)
  - Position management (close one, close all)
  - System control (pause/resume trading, regime override)
  - Status queries (positions, account, regime)

Runs in a background thread, communicates with NT via the Alpaca TradingClient.
"""

from __future__ import annotations

import threading
from decimal import Decimal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest


class TradeRequest(BaseModel):
    symbol: str
    side: str  # "buy" or "sell"
    qty: int
    order_type: str = "market"  # "market" or "limit"
    limit_price: float | None = None


class CloseRequest(BaseModel):
    symbol: str


app = FastAPI(title="KRYOR Trade Engine API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_client: TradingClient | None = None
_trading_paused = False


def set_client(client: TradingClient) -> None:
    global _client
    _client = client


# ── Trade Endpoints ────────────────────────────────────────


@app.post("/api/trade")
def submit_trade(req: TradeRequest):
    if _client is None:
        raise HTTPException(503, "Trading client not initialized")
    if _trading_paused:
        raise HTTPException(403, "Trading is paused")

    side = OrderSide.BUY if req.side.lower() == "buy" else OrderSide.SELL

    try:
        if req.order_type == "limit" and req.limit_price:
            order = _client.submit_order(LimitOrderRequest(
                symbol=req.symbol,
                qty=req.qty,
                side=side,
                type=OrderType.LIMIT,
                time_in_force=TimeInForce.DAY,
                limit_price=round(req.limit_price, 2),
            ))
        else:
            order = _client.submit_order(MarketOrderRequest(
                symbol=req.symbol,
                qty=req.qty,
                side=side,
                type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY,
            ))
        return {
            "status": "submitted",
            "order_id": str(order.id),
            "symbol": req.symbol,
            "side": req.side,
            "qty": req.qty,
            "order_status": order.status.value,
        }
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/close")
def close_position(req: CloseRequest):
    if _client is None:
        raise HTTPException(503, "Trading client not initialized")
    try:
        _client.close_position(req.symbol)
        return {"status": "closed", "symbol": req.symbol}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/close-all")
def close_all():
    if _client is None:
        raise HTTPException(503, "Trading client not initialized")
    try:
        _client.close_all_positions(cancel_orders=True)
        return {"status": "all_closed"}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/cancel-all")
def cancel_all_orders():
    if _client is None:
        raise HTTPException(503, "Trading client not initialized")
    try:
        _client.cancel_orders()
        return {"status": "all_cancelled"}
    except Exception as e:
        raise HTTPException(400, str(e))


# ── System Control ─────────────────────────────────────────


@app.post("/api/pause")
def pause_trading():
    global _trading_paused
    _trading_paused = True
    return {"status": "paused"}


@app.post("/api/resume")
def resume_trading():
    global _trading_paused
    _trading_paused = False
    return {"status": "resumed"}


# ── Status Queries ─────────────────────────────────────────


@app.get("/api/status")
def get_status():
    if _client is None:
        raise HTTPException(503, "Trading client not initialized")
    try:
        account = _client.get_account()
        positions = _client.get_all_positions()
        return {
            "paused": _trading_paused,
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "positions": [
                {
                    "symbol": p.symbol,
                    "side": p.side,
                    "qty": int(p.qty),
                    "avg_entry": float(p.avg_entry_price),
                    "current_price": float(p.current_price),
                    "unrealized_pnl": float(p.unrealized_pl),
                    "pnl_pct": float(p.unrealized_plpc) * 100,
                }
                for p in positions
            ],
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/positions")
def get_positions():
    if _client is None:
        raise HTTPException(503, "Trading client not initialized")
    try:
        positions = _client.get_all_positions()
        return [
            {
                "symbol": p.symbol,
                "side": p.side,
                "qty": int(p.qty),
                "avg_entry": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "unrealized_pnl": float(p.unrealized_pl),
                "market_value": float(p.market_value),
            }
            for p in positions
        ]
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/orders")
def get_orders():
    if _client is None:
        raise HTTPException(503, "Trading client not initialized")
    try:
        orders = _client.get_orders()
        return [
            {
                "id": str(o.id),
                "symbol": o.symbol,
                "side": o.side.value,
                "qty": str(o.qty),
                "type": o.type.value,
                "status": o.status.value,
                "limit_price": str(o.limit_price) if o.limit_price else None,
                "filled_avg_price": str(o.filled_avg_price) if o.filled_avg_price else None,
                "created_at": str(o.created_at),
            }
            for o in orders
        ]
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/health")
def health():
    return {"status": "ok", "paused": _trading_paused}


def start_api_server(client: TradingClient, port: int = 8001) -> threading.Thread:
    """Start the API server in a background daemon thread."""
    set_client(client)
    thread = threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={"host": "0.0.0.0", "port": port, "log_level": "warning"},
        daemon=True,
    )
    thread.start()
    return thread
