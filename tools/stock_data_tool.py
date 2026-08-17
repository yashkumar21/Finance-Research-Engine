"""Finnhub-backed tool for the Research Agent."""

import os
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"


def _resolve_symbol(query: str, api_key: str) -> Optional[str]:
    """Best-effort lookup of a ticker symbol for a company name via Finnhub's /search.

    Only called as a fallback when a literal quote lookup for the query fails,
    so a plain ticker like "AAPL" never pays for this extra request. Prefers
    plain US-listed common stock (no exchange suffix like ".TO" or ".SW") over
    other listings of the same company; falls back to the top raw result.
    """
    try:
        resp = requests.get(
            f"{FINNHUB_BASE_URL}/search",
            params={"q": query, "token": api_key},
            timeout=10,
        )
        resp.raise_for_status()
        results = (resp.json() or {}).get("result") or []
    except requests.exceptions.RequestException:
        return None

    preferred = [
        r
        for r in results
        if r.get("type") == "Common Stock" and "." not in (r.get("symbol") or "")
    ]
    for r in preferred or results:
        symbol = r.get("symbol")
        if symbol:
            return symbol
    return None


def _fetch_quote(ticker: str, api_key: str) -> dict:
    resp = requests.get(
        f"{FINNHUB_BASE_URL}/quote",
        params={"symbol": ticker, "token": api_key},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json() or {}


def get_stock_data(ticker: str) -> dict:
    """Fetch quantitative stock data for a given ticker symbol or company name.

    Retrieves the latest price, trailing P/E ratio, and year-over-year revenue
    growth rate using the Finnhub API. If the input isn't a literal ticker
    (e.g. "Apple" instead of "AAPL"), falls back to a symbol search and
    resolves it before fetching data. This tool performs no qualitative
    analysis and makes no buy/sell recommendations - it only returns raw
    numeric data.

    Args:
        ticker: A stock ticker symbol or company name, e.g. "AAPL", "MSFT",
            "SPY", or "Apple".

    Returns:
        On success, a dict with keys:
            success (bool): True
            ticker (str): the resolved ticker symbol, uppercased
            price (float): most recent price
            pe_ratio (float or None): trailing P/E ratio, if available
            revenue_growth (float or None): YoY revenue growth rate (e.g. 0.0868
                for 8.68%), if available
        On failure (invalid ticker/name, no data found, or a network/API error), a
        dict with keys:
            success (bool): False
            ticker (str): the ticker symbol or name as given, uppercased
            error (str): a human-readable description of what went wrong
    """
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return {"success": False, "ticker": ticker, "error": "Ticker symbol must not be empty."}

    api_key = os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        return {"success": False, "ticker": ticker, "error": "FINNHUB_API_KEY is not set."}

    try:
        quote = _fetch_quote(ticker, api_key)

        if not quote.get("t"):
            resolved = _resolve_symbol(ticker, api_key)
            if resolved and resolved != ticker:
                ticker = resolved
                quote = _fetch_quote(ticker, api_key)

        if not quote.get("t"):
            return {
                "success": False,
                "ticker": ticker,
                "error": f"No price data found for ticker '{ticker}'. It may be invalid or delisted.",
            }

        price = float(quote["c"])

        metric_resp = requests.get(
            f"{FINNHUB_BASE_URL}/stock/metric",
            params={"symbol": ticker, "metric": "all", "token": api_key},
            timeout=10,
        )
        metric_resp.raise_for_status()
        metric = (metric_resp.json() or {}).get("metric") or {}

        pe_ratio = metric.get("peTTM")
        revenue_growth = metric.get("revenueGrowthTTMYoy")

        return {
            "success": True,
            "ticker": ticker,
            "price": price,
            "pe_ratio": float(pe_ratio) if isinstance(pe_ratio, (int, float)) else None,
            "revenue_growth": float(revenue_growth) / 100
            if isinstance(revenue_growth, (int, float))
            else None,
        }

    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status == 429:
            return {
                "success": False,
                "ticker": ticker,
                "error": "Finnhub API rate limit exceeded. Please try again shortly.",
            }
        return {
            "success": False,
            "ticker": ticker,
            "error": f"Finnhub API error ({status}) while fetching data for '{ticker}'.",
        }
    except requests.exceptions.RequestException as exc:
        return {
            "success": False,
            "ticker": ticker,
            "error": f"Network error while fetching data for '{ticker}': {exc}",
        }
    except Exception as exc:
        return {
            "success": False,
            "ticker": ticker,
            "error": f"Failed to fetch data for '{ticker}': {exc}",
        }
