"""Jev (TypeSafe AI) screening tool, called through TypeSafe's own API.

Jev is a "System 1" decision model: instead of generating text it answers a
fixed set of typed questions about a state directly, as probabilities, in a
single call. The screener uses it to cheaply decide which tickers deserve a
full (expensive) Gemini research brief.

Provider history: Vercel AI Gateway (typesafe-ai/jev) until 2026-09-26, when
Vercel restricted Jev to paid credits; then Requesty (typesafe/jev-1.13.0),
where the held-out results reproduced (eval/results/
compare-2026-09-27T070245Z.md); now TypeSafe directly, same pinned model.
"""

import os
import time

import requests
from dotenv import load_dotenv

from pricing import jev_cost
from schemas import ScreenResult, Usage

load_dotenv()

SYSTEMONE_URL = "https://api.typesafe.ai/v1/systemone"
# Pinned rather than jev-latest so results stay comparable across runs.
JEV_MODEL = "jev-1.13.0"
MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 1.0
# Rate limits and overload (TypeSafe's 529) plus transient upstream errors -
# Jev returned frequent 503s in its first weeks of early access.
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504, 529}

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
    # Chosen with eval/tune_questions.py: "would want a full brief" returned
    # ~0.67 for every ticker; "specific new development" wordings fire for
    # nearly every large cap. This wording spreads with a sensible base rate.
    "needs_analysis": {
        "type": "noul",
        "instructions": (
            "An analyst who already covers this company would need to update their "
            "view of it because of these headlines"
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


def _post_with_retry(payload: dict, api_key: str) -> tuple[requests.Response, int, float]:
    """POST to /v1/systemone, retrying rate limits, overload and transient errors.

    Returns (response, attempts made, latency in ms of the final attempt) -
    latency excludes failed attempts and backoff, so it reflects the model's
    speed rather than upstream availability, which attempts captures.
    """
    for attempt in range(1, MAX_RETRIES + 2):
        started = time.perf_counter()
        resp = requests.post(
            SYSTEMONE_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        if resp.status_code not in RETRYABLE_STATUS_CODES or attempt > MAX_RETRIES:
            return resp, attempt, latency_ms
        time.sleep(RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)))
    return resp, attempt, latency_ms


def screen_ticker(ticker: str, quote: dict, news: dict, questions: dict = QUESTIONS) -> dict:
    """Screen one ticker with Jev.

    Args:
        ticker: The ticker symbol, e.g. "AAPL".
        quote: A get_quote_snapshot result (uses "pct_change").
        news: A get_company_news result (uses "headlines").
        questions: Override the question set (the eval uses this to test
            rewordings). Must keep the sentiment, material_event and
            needs_analysis keys and their types.

    Returns:
        On success, a ScreenResult dict: Jev's sentiment choice with its
        probabilities and confidence, material_event and needs_analysis
        probabilities (0-1), the model that answered, wall-clock latency, and
        usage with the per-call cost (the provider's if it reports one,
        otherwise tokens x Jev's list price, flagged as an estimate).
        On failure (missing key, network/API/rate-limit error, or an
        unexpected response shape), a dict with keys success (False),
        ticker, and error.
    """
    ticker = (ticker or "").strip().upper()

    api_key = os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        return {"success": False, "ticker": ticker, "error": "TYPESAFE_API_KEY is not set."}

    payload = {
        "model": JEV_MODEL,
        "state": build_state(ticker, quote, news),
        "questions": questions,
    }

    try:
        resp, attempts, latency_ms = _post_with_retry(payload, api_key)

        if resp.status_code == 429:
            return {"success": False, "ticker": ticker, "error": "Jev rate limit exceeded after retries."}
        if not resp.ok:
            # TypeSafe errors are {"detail": {"error_type", "message"}}; also
            # handle {"error": {...}} and a flat {"message"} from proxies.
            try:
                error_body = resp.json()
                nested = next(
                    (error_body[k] for k in ("detail", "error") if isinstance(error_body.get(k), dict)), {}
                )
                message = nested.get("message") or error_body.get("message") or resp.text
            except ValueError:
                message = resp.text
            return {
                "success": False,
                "ticker": ticker,
                "error": f"Jev API error ({resp.status_code}) after {attempts} attempt(s): {message}",
                # Lets the scanner tell account problems (401/402/403),
                # which hit every ticker, from one-off failures.
                "status_code": resp.status_code,
            }

        body = resp.json()
        answers = body["answers"]
        usage = body.get("usage") or {}
        input_tokens = int(usage.get("input_tokens") or 0)
        # TypeSafe's docs show no cost field; use one if it ever appears.
        reported_cost = usage.get("cost", body.get("cost"))

        result = ScreenResult(
            ticker=ticker,
            sentiment=answers["sentiment"]["choice"],
            sentiment_probabilities=answers["sentiment"].get("probabilities") or {},
            sentiment_confidence=float(answers["sentiment"]["confidence"]),
            material_event=float(answers["material_event"]["noul"]),
            needs_analysis=float(answers["needs_analysis"]["noul"]),
            model=body.get("model", JEV_MODEL),
            latency_ms=round(latency_ms, 1),
            attempts=attempts,
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=int(usage.get("output_tokens") or 0),
                cost_usd=float(reported_cost) if reported_cost is not None else jev_cost(input_tokens),
                cost_is_estimate=reported_cost is None,
                billed_cost_usd=float(reported_cost) if reported_cost is not None else None,
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
