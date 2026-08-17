"""Finnhub-backed tool for the Research Agent."""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"


def get_stock_data(ticker: str) -> dict:
    """Fetch quantitative stock data for a given ticker symbol.

    Retrieves the latest price, trailing P/E ratio, and year-over-year revenue
    growth rate for the given ticker using the Finnhub API. This tool performs
    no qualitative analysis and makes no buy/sell recommendations - it only
    returns raw numeric data.

    Args:
        ticker: The stock ticker symbol to look up, e.g. "AAPL", "MSFT", "SPY".

    Returns:
        On success, a dict with keys:
            success (bool): True
            ticker (str): the ticker symbol, uppercased
            price (float): most recent price
            pe_ratio (float or None): trailing P/E ratio, if available
            revenue_growth (float or None): YoY revenue growth rate (e.g. 0.0868
                for 8.68%), if available
        On failure (invalid ticker, no data found, or a network/API error), a
        dict with keys:
            success (bool): False
            ticker (str): the ticker symbol, uppercased
            error (str): a human-readable description of what went wrong
    """
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return {"success": False, "ticker": ticker, "error": "Ticker symbol must not be empty."}

    api_key = os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        return {"success": False, "ticker": ticker, "error": "FINNHUB_API_KEY is not set."}

    try:
        quote_resp = requests.get(
            f"{FINNHUB_BASE_URL}/quote",
            params={"symbol": ticker, "token": api_key},
            timeout=10,
        )
        quote_resp.raise_for_status()
        quote = quote_resp.json() or {}

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
