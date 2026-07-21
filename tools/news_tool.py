"""Finnhub-backed news tool for the Sentiment Agent."""

import os
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv

load_dotenv()

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
LOOKBACK_DAYS = 7
MAX_HEADLINES = 10


def get_company_news(ticker: str) -> dict:
    """Fetch recent news headlines for a stock ticker.

    Retrieves headlines from the last 7 days for the given ticker using the
    Finnhub API. This tool performs no sentiment analysis or judgment of any
    kind - it only returns the raw headlines, their sources, and URLs so a
    caller can read and cite them directly.

    Args:
        ticker: The stock ticker symbol to look up, e.g. "AAPL", "MSFT".

    Returns:
        On success, a dict with keys:
            success (bool): True
            ticker (str): the ticker symbol, uppercased
            headlines (list[dict]): each with "headline", "source", "url",
                and "published_at" (ISO 8601 string), most recent first.
                An empty list means no recent news coverage was found - this
                is still a success, not an error.
            note (str, optional): present only when there is no recent
                coverage, explaining why the list is empty.
        On failure (invalid ticker, or a network/API/rate-limit error), a
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
        profile_resp = requests.get(
            f"{FINNHUB_BASE_URL}/stock/profile2",
            params={"symbol": ticker, "token": api_key},
            timeout=10,
        )
        profile_resp.raise_for_status()
        profile = profile_resp.json()

        if not profile:
            return {
                "success": False,
                "ticker": ticker,
                "error": f"No company profile found for ticker '{ticker}'. It may be invalid.",
            }

        today = datetime.now(timezone.utc).date()
        start = today - timedelta(days=LOOKBACK_DAYS)

        news_resp = requests.get(
            f"{FINNHUB_BASE_URL}/company-news",
            params={
                "symbol": ticker,
                "from": start.isoformat(),
                "to": today.isoformat(),
                "token": api_key,
            },
            timeout=10,
        )
        news_resp.raise_for_status()
        articles = news_resp.json() or []

        headlines = []
        for article in articles:
            text = article.get("headline")
            if not text:
                continue
            timestamp = article.get("datetime")
            published_at = (
                datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
                if timestamp
                else ""
            )
            headlines.append(
                {
                    "headline": text,
                    "source": article.get("source", ""),
                    "url": article.get("url", ""),
                    "published_at": published_at,
                }
            )
        headlines = headlines[:MAX_HEADLINES]

        result = {"success": True, "ticker": ticker, "headlines": headlines}
        if not headlines:
            result["note"] = (
                f"No recent news coverage found for '{ticker}' in the last {LOOKBACK_DAYS} days."
            )
        return result

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
            "error": f"Finnhub API error ({status}) while fetching news for '{ticker}'.",
        }
    except requests.exceptions.RequestException as exc:
        return {
            "success": False,
            "ticker": ticker,
            "error": f"Network error while fetching news for '{ticker}': {exc}",
        }
    except Exception as exc:
        return {
            "success": False,
            "ticker": ticker,
            "error": f"Unexpected error while fetching news for '{ticker}': {exc}",
        }
