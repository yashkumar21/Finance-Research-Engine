"""Finnhub-backed news tool for the Sentiment Agent and the scanner."""

import os
import re
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv

load_dotenv()

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
LOOKBACK_DAYS = 7
MAX_HEADLINES = 10

# Legal suffixes dropped from Finnhub's company name before matching it in
# article text ("NVIDIA Corp" -> "nvidia").
_NAME_SUFFIXES = {
    "inc", "corp", "corporation", "co", "company", "ltd", "plc", "llc", "lp", "holdings", "holding",
    "group", "the", "class", "a", "b", "c", "sa", "nv", "ag", "se", "incorporated", "limited",
}
# First words too generic to identify a company on their own ("Bank of
# America" must match in full, not as "bank").
_GENERIC_FIRST_WORDS = {
    "bank", "american", "general", "first", "united", "international", "national", "global",
    "public", "southern", "western", "eastern", "northern", "digital", "capital",
}


def _name_patterns(ticker: str, company_name: str) -> list[re.Pattern]:
    """Patterns that mean an article is about this company, not just tagged with it."""
    words = [w for w in re.findall(r"[a-z0-9&']+", company_name.lower()) if w not in _NAME_SUFFIXES]
    patterns = []
    if words:
        patterns.append(re.compile(r"\b" + r"\s+".join(map(re.escape, words)) + r"\b", re.IGNORECASE))
        if len(words) > 1 and len(words[0]) >= 5 and words[0] not in _GENERIC_FIRST_WORDS:
            patterns.append(re.compile(r"\b" + re.escape(words[0]) + r"\b", re.IGNORECASE))
    # Tickers like A, T or ON are ordinary words, so short ones only count in
    # ticker notation: "(T)", "NYSE:T", "$T".
    if len(ticker) >= 3:
        patterns.append(re.compile(r"\b" + re.escape(ticker) + r"\b"))
    patterns.append(re.compile(r"(\(|:|\$)" + re.escape(ticker) + r"\b"))
    return patterns


def _is_about(article: dict, patterns: list[re.Pattern]) -> bool:
    text = f"{article.get('headline', '')} {article.get('summary', '')}"
    return any(p.search(text) for p in patterns)


def headlines_about(headlines: list[dict], ticker: str, company_name: str) -> list[dict]:
    """The headlines that name the company or its ticker (headline text only)."""
    patterns = _name_patterns(ticker, company_name or "")
    return [h for h in headlines if _is_about(h, patterns)]


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
            company_name (str or None): the company's name per Finnhub
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
    return _fetch_news(ticker, relevant_only=False)


def get_relevant_company_news(ticker: str) -> dict:
    """get_company_news, keeping only articles that mention the company.

    Finnhub's company-news feed also returns market roundups and other
    companies' stories tagged with the ticker (for NVDA: Tesla deliveries,
    Medtronic's dividend). This keeps articles whose headline or summary
    names the company or its ticker, then takes the most recent
    MAX_HEADLINES. If none qualify, headlines is empty with a note saying so.

    Used by the research brief. The scanner deliberately keeps the
    unfiltered get_company_news: its Jev policy was evaluated on that input.
    """
    return _fetch_news(ticker, relevant_only=True)


def _fetch_news(ticker: str, relevant_only: bool) -> dict:
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
        fetched = len(articles)
        if relevant_only:
            patterns = _name_patterns(ticker, profile.get("name") or "")
            articles = [a for a in articles if _is_about(a, patterns)]

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

        result = {"success": True, "ticker": ticker, "headlines": headlines, "company_name": profile.get("name")}
        if not headlines and relevant_only and fetched:
            result["note"] = (
                f"Finnhub returned {fetched} articles for '{ticker}' in the last {LOOKBACK_DAYS} days, "
                "but none mention the company by name or ticker."
            )
        elif not headlines:
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
