"""yfinance-backed tool for the Research Agent."""

import yfinance as yf


def get_stock_data(ticker: str) -> dict:
    """Fetch quantitative stock data for a given ticker symbol.

    Retrieves the latest closing price, trailing P/E ratio, and revenue growth
    rate for the given ticker using Yahoo Finance data. This tool performs no
    qualitative analysis and makes no buy/sell recommendations - it only
    returns raw numeric data.

    Args:
        ticker: The stock ticker symbol to look up, e.g. "AAPL", "MSFT", "SPY".

    Returns:
        On success, a dict with keys:
            success (bool): True
            ticker (str): the ticker symbol, uppercased
            price (float): most recent closing price
            pe_ratio (float or None): trailing P/E ratio, if available
            revenue_growth (float or None): revenue growth rate, if available
        On failure (invalid ticker, no data found, or a network/API error), a
        dict with keys:
            success (bool): False
            ticker (str): the ticker symbol, uppercased
            error (str): a human-readable description of what went wrong
    """
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return {"success": False, "ticker": ticker, "error": "Ticker symbol must not be empty."}

    try:
        yf_ticker = yf.Ticker(ticker)
        history = yf_ticker.history(period="5d")

        if history.empty:
            return {
                "success": False,
                "ticker": ticker,
                "error": f"No price history found for ticker '{ticker}'. It may be invalid or delisted.",
            }

        price = float(history["Close"].iloc[-1])

        try:
            info = yf_ticker.info or {}
        except Exception:
            info = {}

        pe_ratio = info.get("trailingPE")
        revenue_growth = info.get("revenueGrowth")

        return {
            "success": True,
            "ticker": ticker,
            "price": price,
            "pe_ratio": float(pe_ratio) if isinstance(pe_ratio, (int, float)) else None,
            "revenue_growth": float(revenue_growth) if isinstance(revenue_growth, (int, float)) else None,
        }

    except Exception as exc:
        return {
            "success": False,
            "ticker": ticker,
            "error": f"Failed to fetch data for '{ticker}': {exc}",
        }
