"""Real-network tests for tools.stock_data_tool.get_stock_data.

No mocking - these hit the Finnhub API directly, per design.
"""

import pytest

from tools.stock_data_tool import get_stock_data


def _assert_structure(result: dict):
    assert isinstance(result, dict)
    assert "success" in result
    assert "ticker" in result
    assert isinstance(result["ticker"], str)
    if result["success"]:
        assert set(result.keys()) == {"success", "ticker", "price", "pe_ratio", "revenue_growth"}
        assert isinstance(result["price"], float)
        assert result["pe_ratio"] is None or isinstance(result["pe_ratio"], float)
        assert result["revenue_growth"] is None or isinstance(result["revenue_growth"], float)
    else:
        assert set(result.keys()) == {"success", "ticker", "error"}
        assert isinstance(result["error"], str) and result["error"]


def test_aapl_happy_path():
    result = get_stock_data("AAPL")
    _assert_structure(result)
    assert result["success"] is True
    assert result["ticker"] == "AAPL"
    assert result["price"] > 0
    assert result["pe_ratio"] is not None
    assert result["revenue_growth"] is not None


def test_spy_partial_data_still_success():
    result = get_stock_data("SPY")
    _assert_structure(result)
    assert result["success"] is True
    assert result["ticker"] == "SPY"
    assert result["price"] > 0
    # ETFs typically lack a trailing P/E / revenue growth - missing fields
    # must not turn the whole call into an error.


def test_brk_b_dash_ticker():
    result = get_stock_data("BRK-B")
    _assert_structure(result)
    assert result["success"] is True
    assert result["ticker"] == "BRK-B"
    assert result["price"] > 0


def test_invalid_ticker_returns_structured_error_not_exception():
    result = get_stock_data("ZZZZZZINVALID")
    _assert_structure(result)
    assert result["success"] is False
    assert result["ticker"] == "ZZZZZZINVALID"
    assert "error" in result


def test_empty_ticker_returns_structured_error():
    result = get_stock_data("")
    _assert_structure(result)
    assert result["success"] is False
