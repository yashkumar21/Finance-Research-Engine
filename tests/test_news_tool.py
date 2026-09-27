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


# --- get_relevant_company_news: headlines about the company, not just tagged with it ---

from tools.news_tool import _is_about, _name_patterns, get_relevant_company_news  # noqa: E402


def _about(ticker, name, headline, summary=""):
    return _is_about({"headline": headline, "summary": summary}, _name_patterns(ticker, name))


def test_company_name_matches_without_legal_suffix():
    assert _about("NVDA", "NVIDIA Corp", "3 Reasons Why Nvidia Fits Warren Buffett's Style")
    assert not _about("NVDA", "NVIDIA Corp", "Tesla Reports Q3 Deliveries in Early October")


def test_summary_counts_as_well_as_headline():
    assert _about("NVDA", "NVIDIA Corp", "Chip stocks rally", summary="Nvidia led the gains.")


def test_generic_first_word_needs_the_full_name():
    assert _about("BAC", "Bank of America Corp", "Bank of America Stock Trades 13% Below Its High")
    assert not _about("BAC", "Bank of America Corp", "Bank stocks slide on rate fears")


def test_short_tickers_only_match_in_ticker_notation():
    # "ON" and "T" are ordinary words; only "(ON)", "NYSE:T", "$T" count.
    assert _about("ON", "ON Semiconductor Corp", "ON Semiconductor (ON) Stock Stays Near Fair Value")
    assert not _about("ON", "Fake Co", "Stocks on the move on Friday")
    assert _about("T", "AT&T Inc", "Does AT&T's Dividend Reaffirmation Signal Stability?")
    assert not _about("T", "Fake Co", "T-Mobile adds subscribers")


def test_relevant_news_live_nvda():
    result = get_relevant_company_news("NVDA")
    _assert_structure(result)
    assert result["success"] is True
    # Finnhub may not return summaries through this tool, so allow a
    # headline that matched on its summary - but most should name Nvidia.
    about = [h for h in result["headlines"] if "nvidia" in h["headline"].lower() or "NVDA" in h["headline"]]
    assert len(about) >= len(result["headlines"]) // 2
