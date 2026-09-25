"""Tests for tools.rate_limit.TokenBucket - no network, fast rates."""

import asyncio
import time

import pytest

from tools.rate_limit import TokenBucket


@pytest.mark.asyncio
async def test_full_bucket_does_not_wait():
    bucket = TokenBucket(rate_per_minute=60, capacity=5)
    started = time.monotonic()
    await bucket.acquire(5)
    assert time.monotonic() - started < 0.05


@pytest.mark.asyncio
async def test_empty_bucket_waits_for_refill():
    bucket = TokenBucket(rate_per_minute=600, capacity=2)  # 10 tokens/s
    await bucket.acquire(2)
    started = time.monotonic()
    await bucket.acquire(2)
    assert 0.15 <= time.monotonic() - started < 0.4


@pytest.mark.asyncio
async def test_concurrent_callers_share_the_rate():
    bucket = TokenBucket(rate_per_minute=1200, capacity=1)  # 20 tokens/s
    started = time.monotonic()
    await asyncio.gather(*(bucket.acquire(1) for _ in range(11)))
    # First token is free, the other 10 need 0.5s of refill between them.
    assert 0.45 <= time.monotonic() - started < 0.8


@pytest.mark.asyncio
async def test_acquiring_more_than_capacity_is_an_error():
    with pytest.raises(ValueError):
        await TokenBucket(capacity=5).acquire(6)
