"""Tests for screener.scan.run_scan.

Real-network, matching the repo's no-mocking convention: Finnhub for data and
Jev for screening. Skipped without FINNHUB_API_KEY and REQUESTY_API_KEY.
Escalation stays off so no Gemini brief runs.
"""

import os

import pytest
from dotenv import load_dotenv

from schemas import ScanReport
from screener.scan import load_universe, run_scan

load_dotenv()

pytestmark = pytest.mark.skipif(
    not (os.environ.get("FINNHUB_API_KEY") and os.environ.get("REQUESTY_API_KEY")),
    reason="FINNHUB_API_KEY and REQUESTY_API_KEY are required",
)


@pytest.mark.asyncio
async def test_scan_without_escalation_screens_every_ticker():
    tickers = ["AAPL", "MSFT", "ZZZZZZINVALID"]
    report = await run_scan(tickers, "custom", escalate=False)

    ScanReport.model_validate(report.model_dump())
    assert [r.ticker for r in report.results] == tickers
    assert report.escalation_enabled is False
    assert report.briefs_generated == 0
    assert all(r.brief is None for r in report.results)

    for r in report.results[:2]:
        assert r.screen is not None, r.screen_error
        assert r.screen.sentiment in {"bullish", "bearish", "neutral"}
        assert r.screen.usage.cost_usd > 0
    assert report.jev_cost_usd > 0
    assert report.jev_model

    # A ticker that can't be fetched is escalated, never silently dropped.
    invalid = report.results[2]
    assert invalid.screen is None
    assert "data fetch failed" in invalid.screen_error
    assert invalid.decision.escalate is True


def test_universes_load():
    sp100, sp500 = load_universe("sp100"), load_universe("sp500")
    assert 95 <= len(sp100) <= 105
    assert 495 <= len(sp500) <= 510
    assert set(sp100) <= set(sp500)
    assert all(t == t.strip().upper() and not t.startswith("#") for t in sp500)
