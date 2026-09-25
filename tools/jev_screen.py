"""Jev (TypeSafe AI) screening tool, called through Vercel AI Gateway.

Jev is a "System 1" decision model: instead of generating text it answers a
fixed set of typed questions about a state directly, as probabilities, in a
single call. The screener uses it to cheaply decide which tickers deserve a
full (expensive) Gemini research brief.
"""

import os
import time

import requests
from dotenv import load_dotenv

from schemas import ScreenResult, Usage

load_dotenv()

GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/typesafe"
JEV_MODEL = "typesafe-ai/jev"
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0

SENTIMENT_CRITERIA = {
    "bullish": "The headlines skew positive for the company",
    "bearish": "The headlines skew negative for the company",
    "neutral": "The headlines are mixed, non-committal, or there is no recent coverage",
}

QUESTIONS = {
    "sentiment": {
        "type": "choice",
        "instructions": "The overall tone of this company's recent news coverage",
        "criteria": SENTIMENT_CRITERIA,
    },
    "material_event": {
        "type": "noul",
        "instructions": (
            "The headlines report a material event for this company: an earnings "
            "surprise, guidance change, merger or acquisition, major litigation, "
            "regulatory action, or executive change"
        ),
    },
    "needs_analysis": {
        "type": "noul",
        "instructions": (
            "A professional equity analyst covering this company would want a full "
            "research brief on it today"
        ),
    },
}


def build_state(ticker: str, quote: dict, news: dict) -> dict:
    """The state Jev screens: today's price move plus recent headlines."""
    return {
        "ticker": ticker,
        "pct_change_today": quote.get("pct_change"),
        "headlines": [
            {"headline": h["headline"], "source": h["source"], "published_at": h["published_at"]}
            for h in news.get("headlines", [])
        ],
    }


def _post_with_retry(payload: dict, api_key: str) -> requests.Response:
    """POST to /systemone, retrying 429s with exponential backoff."""
    for attempt in range(MAX_RETRIES + 1):
        resp = requests.post(
            f"{GATEWAY_BASE_URL}/v1/systemone",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        if resp.status_code != 429 or attempt == MAX_RETRIES:
            return resp
        time.sleep(RETRY_BACKOFF_SECONDS * (2**attempt))
    return resp


def screen_ticker(ticker: str, quote: dict, news: dict) -> dict:
    """Screen one ticker with Jev.

    Args:
        ticker: The ticker symbol, e.g. "AAPL".
        quote: A get_quote_snapshot result (uses "pct_change").
        news: A get_company_news result (uses "headlines").

    Returns:
        On success, a ScreenResult dict: Jev's sentiment choice with its
        probabilities and confidence, material_event and needs_analysis
        probabilities (0-1), the model that answered, wall-clock latency, and
        usage with the exact per-call cost reported by the gateway.
        On failure (missing key, network/API/rate-limit error, or an
        unexpected response shape), a dict with keys success (False),
        ticker, and error.
    """
    ticker = (ticker or "").strip().upper()

    api_key = os.environ.get("AI_GATEWAY_API_KEY")
    if not api_key:
        return {"success": False, "ticker": ticker, "error": "AI_GATEWAY_API_KEY is not set."}

    payload = {
        "model": JEV_MODEL,
        "state": build_state(ticker, quote, news),
        "questions": QUESTIONS,
    }

    try:
        started = time.perf_counter()
        resp = _post_with_retry(payload, api_key)
        latency_ms = (time.perf_counter() - started) * 1000

        if resp.status_code == 429:
            return {"success": False, "ticker": ticker, "error": "Jev rate limit exceeded after retries."}
        if not resp.ok:
            # TypeSafe errors are {"message", "error_type"}; gateway-level
            # errors (auth, billing) are {"error": {"message", "type"}}.
            try:
                error_body = resp.json()
                nested = error_body.get("error") if isinstance(error_body.get("error"), dict) else {}
                message = nested.get("message") or error_body.get("message") or resp.text
            except ValueError:
                message = resp.text
            return {
                "success": False,
                "ticker": ticker,
                "error": f"Jev API error ({resp.status_code}): {message}",
            }

        body = resp.json()
        answers = body["answers"]
        usage = body.get("usage") or {}
        gateway = (body.get("provider_metadata") or {}).get("gateway") or {}

        result = ScreenResult(
            ticker=ticker,
            sentiment=answers["sentiment"]["choice"],
            sentiment_probabilities=answers["sentiment"].get("probabilities") or {},
            sentiment_confidence=float(answers["sentiment"]["confidence"]),
            material_event=float(answers["material_event"]["noul"]),
            needs_analysis=float(answers["needs_analysis"]["noul"]),
            model=body.get("model", JEV_MODEL),
            latency_ms=round(latency_ms, 1),
            usage=Usage(
                input_tokens=int(usage.get("input_tokens") or 0),
                output_tokens=int(usage.get("output_tokens") or 0),
                # marketCost is the list price of the call; cost is what was
                # billed, which is 0 while free gateway credits cover it.
                cost_usd=float(gateway.get("marketCost") or gateway.get("cost") or 0),
                billed_cost_usd=float(gateway.get("cost") or 0),
            ),
        )
        return result.model_dump()

    except requests.exceptions.RequestException as exc:
        return {"success": False, "ticker": ticker, "error": f"Network error calling Jev: {exc}"}
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "success": False,
            "ticker": ticker,
            "error": f"Unexpected Jev response shape: {exc!r}",
        }
