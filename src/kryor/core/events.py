from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Callable, Coroutine

from loguru import logger

EventHandler = Callable[..., Coroutine[Any, Any, None]]


class EventBus:
    """Simple async pub/sub event bus for inter-component communication."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        self._handlers[event_type].append(handler)
        logger.debug(f"Subscribed {handler.__qualname__} to '{event_type}'")

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        self._handlers[event_type].remove(handler)

    async def publish(self, event_type: str, data: Any = None) -> None:
        handlers = self._handlers.get(event_type, [])
        if not handlers:
            return
        tasks = [asyncio.create_task(h(data)) for h in handlers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Event handler error for '{event_type}': {result}")


# Event type constants
BAR_UPDATE = "bar.update"
SIGNAL_NEW = "signal.new"
ORDER_FILLED = "order.filled"
REGIME_UPDATE = "regime.update"
CIRCUIT_BREAKER = "circuit_breaker.triggered"
