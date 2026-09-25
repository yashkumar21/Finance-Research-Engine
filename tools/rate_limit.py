"""Shared async token bucket for Finnhub calls.

Finnhub's free tier allows 60 calls/min; the scanner makes every Finnhub call
through one bucket refilling at 55/min for headroom. The bucket is kept small
so it never bursts: a full bucket plus a minute of refill must not exceed 60
calls in any 60-second window (capacity 5 + 55 refilled = 60).
"""

import asyncio
import time

FINNHUB_CALLS_PER_MINUTE = 55


class TokenBucket:
    def __init__(self, rate_per_minute: float = FINNHUB_CALLS_PER_MINUTE, capacity: int = 5):
        self.rate_per_second = rate_per_minute / 60
        self.capacity = capacity
        self.tokens = float(capacity)
        self.updated = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate_per_second)
        self.updated = now

    async def acquire(self, n: int = 1) -> None:
        """Wait until n calls may be made. n must not exceed capacity."""
        if n > self.capacity:
            raise ValueError(f"Cannot acquire {n} tokens from a bucket of capacity {self.capacity}")
        # The lock makes waiters queue in order, so one large request can't
        # be starved by a stream of small ones.
        async with self._lock:
            self._refill()
            while self.tokens < n:
                await asyncio.sleep((n - self.tokens) / self.rate_per_second)
                self._refill()
            self.tokens -= n
