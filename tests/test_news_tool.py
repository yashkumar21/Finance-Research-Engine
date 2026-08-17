"""Tests for tools.news_tool.get_company_news.

Real-network tests against Finnhub for the happy path, low/no-coverage
case, and invalid ticker - matching the stock data tool test convention.
Rate-limit and network-failure handling is instead simulated by
monkeypatching requests.get: deliberately hammering Finnhub's free tier
(60 calls/min) to trigger a real 429 would be slow, flaky, and would burn
the account's shared quota, so those two failure modes are exercised
deterministically instead.
"""

import requests

from tools import news_tool
from tools.news_tool import get_company_news


def _assert_structure(result: dict):
    assert isinstance(result, dict)
    assert "success" in result
    assert "ticker" in result
    assert isinstance(result["ticker"], str)
    if result["success"]:
        assert "headlines" in result
        assert isinstance(result["headlines"], list)
        for h in result["headlines"]:
            assert set(h.keys()) == {"headline", "source", "url", "published_at"}
            assert isinstance(h["headline"], str) and h["headline"]
    else:
        assert set(result.keys()) == {"success", "ticker", "error"}
        assert isinstance(result["error"], str) and result["error"]


def test_aapl_happy_path():
    result = get_company_news("AAPL")
    _assert_structure(result)
    assert result["success"] is True
    assert result["ticker"] == "AAPL"
    assert len(result["headlines"]) > 0


def test_obscure_ticker_low_or_no_coverage():
    # UAMY (US Antimony Corp) is a real, thinly-traded micro-cap: low news
    # volume most days. We don't assert an exact headline count (that would
    # be flaky if it happens to have coverage on test day) - only that the
    # tool handles sparse/no coverage gracefully as a success, not an error.
    result = get_company_news("UAMY")
    _assert_structure(result)
    assert result["success"] is True
    assert result["ticker"] == "UAMY"
    if not result["headlines"]:
        assert "note" in result and result["note"]


def test_invalid_ticker_returns_structured_error_not_exception():
    result = get_company_news("ZZZZZZINVALID")
    _assert_structure(result)
    assert result["success"] is False
    assert result["ticker"] == "ZZZZZZINVALID"


def test_empty_ticker_returns_structured_error():
    result = get_company_news("")
    _assert_structure(result)
    assert result["success"] is False


def test_rate_limit_returns_structured_error(monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "dummy-key-for-mocked-test")

    class FakeResponse:
        status_code = 429

        def raise_for_status(self):
            raise requests.exceptions.HTTPError(response=self)

    monkeypatch.setattr(news_tool.requests, "get", lambda *a, **k: FakeResponse())

    result = get_company_news("AAPL")
    _assert_structure(result)
    assert result["success"] is False
    assert "rate limit" in result["error"].lower()


def test_network_failure_returns_structured_error(monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "dummy-key-for-mocked-test")

    def raise_connection_error(*args, **kwargs):
        raise requests.exceptions.ConnectionError("simulated network failure")

    monkeypatch.setattr(news_tool.requests, "get", raise_connection_error)

    result = get_company_news("AAPL")
    _assert_structure(result)
    assert result["success"] is False
