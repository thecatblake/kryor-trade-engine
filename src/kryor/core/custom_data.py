"""Custom data types for inter-component communication via MessageBus.

NautilusTrader's Data is a Cython abstract class that can't be easily subclassed.
We use plain Python dataclasses and communicate via msgbus.publish() directly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import IntEnum


class RegimeEnum(IntEnum):
    BULL = 0
    NEUTRAL = 1
    BEAR = 2


@dataclass
class RegimeData:
    """Published by RegimeActor, consumed by strategies via msgbus."""

    regime: RegimeEnum
    probability: float
    bull_prob: float
    neutral_prob: float
    bear_prob: float
    duration_days: int = 0
    ts_event: int = 0
    ts_init: int = 0

    def __post_init__(self) -> None:
        if self.ts_event == 0:
            self.ts_event = int(time.time() * 1e9)
        if self.ts_init == 0:
            self.ts_init = self.ts_event

    @property
    def is_bull(self) -> bool:
        return self.regime == RegimeEnum.BULL

    @property
    def is_bear(self) -> bool:
        return self.regime == RegimeEnum.BEAR

    @property
    def kelly_fraction(self) -> float:
        if self.regime == RegimeEnum.BULL:
            return 0.50
        elif self.regime == RegimeEnum.NEUTRAL:
            return 0.25
        return 0.0


@dataclass
class CircuitBreakerData:
    """Published when a circuit breaker level is triggered."""

    level: int
    reason: str
    ts_event: int = 0
    ts_init: int = 0

    def __post_init__(self) -> None:
        if self.ts_event == 0:
            self.ts_event = int(time.time() * 1e9)
        if self.ts_init == 0:
            self.ts_init = self.ts_event
