"""Finnhub-backed tool for the Research Agent."""

import csv
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
UNIVERSES_DIR = Path(__file__).resolve().parent.parent / "data" / "universes"


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


def _resolve_and_quote(query: str, api_key: str) -> tuple[str, dict]:
    """Resolve ``query`` to a ticker symbol and return (ticker, quote dict).

    Tries the literal input as a ticker first; only falls back to the
    company-name search in _resolve_symbol if that fails, so a plain ticker
    like "AAPL" never pays for the extra request.
    """
    ticker = query
    quote = _fetch_quote(ticker, api_key)
    if not quote.get("t"):
        resolved = _resolve_symbol(ticker, api_key)
        if resolved and resolved != ticker:
            ticker = resolved
            quote = _fetch_quote(ticker, api_key)
    return ticker, quote


@lru_cache(maxsize=1)
def load_company_names() -> dict[str, str]:
    """Ticker -> company name from data/universes/*_names.csv (e.g. "Tesla, Inc.")."""
    names = {}
    for path in UNIVERSES_DIR.glob("*_names.csv"):
        rows = (line for line in path.read_text().splitlines() if not line.startswith("#"))
        names.update({row["ticker"]: row["name"] for row in csv.DictReader(rows)})
    return names


def get_company_name(ticker: str) -> str:
    """Best-effort display name: the S&P 500 name list, then Finnhub's company
    profile, then the ticker itself. Never raises."""
    ticker = (ticker or "").strip().upper()
    if ticker in load_company_names():
        return load_company_names()[ticker]
    api_key = os.environ.get("FINNHUB_API_KEY")
    if api_key and ticker:
        try:
            resp = requests.get(
                f"{FINNHUB_BASE_URL}/stock/profile2", params={"symbol": ticker, "token": api_key}, timeout=10
            )
            resp.raise_for_status()
            name = (resp.json() or {}).get("name")
            if name:
                return name
        except (requests.exceptions.RequestException, ValueError):
            pass
    return ticker


def resolve_ticker(query: str) -> str:
    """Best-effort resolve a ticker symbol or company name to its canonical ticker.

    Used by the orchestrator to pin down the ticker before the pipeline runs,
    so the Analyst Agent's brief heading matches the data get_stock_data
    fetches (e.g. "Apple" -> "AAPL"), instead of echoing whatever the user
    literally typed. Returns the uppercased input unchanged if it can't be
    resolved - no FINNHUB_API_KEY, a network error, or no match found - so
    callers still get get_stock_data's normal structured error downstream.
    """
    query = (query or "").strip().upper()
    if not query:
        return query

    api_key = os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        return query

    try:
        ticker, _ = _resolve_and_quote(query, api_key)
        return ticker
    except requests.exceptions.RequestException:
        return query


def get_quote_snapshot(ticker: str) -> dict:
    """Fetch just the latest price and today's % change for a ticker.

    A lighter sibling of get_stock_data for the screener: a single /quote
    call (no /stock/metric, no company-name search), since a nightly scan of
    hundreds of tickers is bound by Finnhub's 60 calls/min free-tier limit.

    Returns:
        On success: {"success": True, "ticker", "price", "pct_change"}, where
        pct_change is today's move in percent (e.g. -2.31) or None.
        On failure: {"success": False, "ticker", "error"}.
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
            return {
                "success": False,
                "ticker": ticker,
                "error": f"No price data found for ticker '{ticker}'. It may be invalid or delisted.",
            }
        pct_change = quote.get("dp")
        return {
            "success": True,
            "ticker": ticker,
            "price": float(quote["c"]),
            "pct_change": float(pct_change) if isinstance(pct_change, (int, float)) else None,
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
            "error": f"Finnhub API error ({status}) while fetching a quote for '{ticker}'.",
        }
    except requests.exceptions.RequestException as exc:
        return {
            "success": False,
            "ticker": ticker,
            "error": f"Network error while fetching a quote for '{ticker}': {exc}",
        }


def _number(value) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def get_stock_data(ticker: str) -> dict:
    """Fetch quantitative stock data for a given ticker symbol or company name.

    Retrieves the latest price and today's move, trailing P/E ratio,
    year-over-year revenue growth, 52-week range and market cap using the
    Finnhub API (one quote call and one metrics call). If the input isn't a literal ticker
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
            pct_change (float or None): today's move in percent (e.g. -2.31)
            week52_high, week52_low (float or None): 52-week price range
            market_cap (float or None): market capitalization in USD
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
        ticker, quote = _resolve_and_quote(ticker, api_key)

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
        market_cap_millions = metric.get("marketCapitalization")

        return {
            "success": True,
            "ticker": ticker,
            "price": price,
            "pe_ratio": float(pe_ratio) if isinstance(pe_ratio, (int, float)) else None,
            "revenue_growth": float(revenue_growth) / 100
            if isinstance(revenue_growth, (int, float))
            else None,
            "pct_change": _number(quote.get("dp")),
            "week52_high": _number(metric.get("52WeekHigh")),
            "week52_low": _number(metric.get("52WeekLow")),
            # Finnhub reports market cap in millions.
            "market_cap": market_cap_millions * 1_000_000
            if isinstance(market_cap_millions, (int, float))
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
