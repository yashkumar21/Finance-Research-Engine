"""Tests for tools.jev_screen.screen_ticker.

Real-network tests against Jev via Requesty, matching the repo's
no-mocking convention; skipped when REQUESTY_API_KEY is not set. Headlines
are hand-written so the expected direction of each answer is unambiguous.
"""

import os

import pytest
from dotenv import load_dotenv

from schemas import ScreenResult
from tools.jev_screen import screen_ticker

load_dotenv()

pytestmark = pytest.mark.skipif(
    not os.environ.get("REQUESTY_API_KEY"), reason="REQUESTY_API_KEY is not set"
)


def _news(*headlines):
    return {
        "success": True,
        "ticker": "TEST",
        "headlines": [
            {"headline": h, "source": "Test Wire", "url": "https://example.com", "published_at": "2026-09-25T12:00:00+00:00"}
            for h in headlines
        ],
    }


def _assert_valid(result: dict):
    assert result["success"] is True, result
    ScreenResult(**result)
    assert result["sentiment"] in {"bullish", "bearish", "neutral"}
    for key in ("sentiment_confidence", "material_event", "needs_analysis"):
        assert 0.0 <= result[key] <= 1.0, key
    assert result["latency_ms"] > 0
    assert result["usage"]["cost_usd"] > 0
    assert result["model"]


def test_clearly_negative_material_news():
    result = screen_ticker(
        "TEST",
        {"pct_change": -11.2},
        _news(
            "TestCorp shares plunge after company slashes full-year guidance",
            "SEC opens investigation into TestCorp accounting practices",
            "TestCorp CFO resigns effective immediately",
        ),
    )
    _assert_valid(result)
    assert result["sentiment"] == "bearish"
    assert result["material_event"] >= 0.5


def test_clearly_positive_news():
    result = screen_ticker(
        "TEST",
        {"pct_change": 6.4},
        _news(
            "TestCorp beats earnings estimates and raises full-year outlook",
            "Analysts upgrade TestCorp after record quarterly revenue",
        ),
    )
    _assert_valid(result)
    assert result["sentiment"] == "bullish"


def test_no_news_is_valid():
    result = screen_ticker("TEST", {"pct_change": 0.2}, _news())
    _assert_valid(result)


def test_missing_key_returns_structured_error(monkeypatch):
    monkeypatch.delenv("REQUESTY_API_KEY")
    result = screen_ticker("TEST", {"pct_change": 0.0}, _news())
    assert result == {"success": False, "ticker": "TEST", "error": "REQUESTY_API_KEY is not set."}
