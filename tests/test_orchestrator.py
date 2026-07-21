"""Tests for agents.orchestrator (Research || Sentiment -> Analyst pipeline).

Two isolated Analyst-only tests seed session state directly (bypassing the
upstream agents entirely) to cheaply and deterministically verify the
Analyst's constraint-following - crafted numbers can't coincidentally
appear the way a real ticker's real numbers might. One end-to-end test
runs the full real pipeline for AAPL.
"""

import json
from datetime import datetime, timezone

import pytest
from dotenv import load_dotenv
from google.genai import types

from google.adk.runners import InMemoryRunner

from agents.analyst_agent import analyst_agent
from agents.orchestrator import run_research_brief
from tools.stock_data_tool import get_stock_data

load_dotenv()

APP_NAME = "finance_research_engine_analyst_test"
USER_ID = "test_user"

DISCLAIMER_PHRASE = "informational research only, not investment advice"
FORBIDDEN_PHRASES = ["you should buy", "you should sell", "strong buy", "strong sell", "price target"]


def _number_appears(text: str, value: float) -> bool:
    candidates = {f"{value:.2f}", f"{value:.1f}", f"{value:g}", str(round(value))}
    return any(c in text for c in candidates)


async def _run_analyst_with_seeded_state(state: dict) -> str:
    runner = InMemoryRunner(agent=analyst_agent, app_name=APP_NAME)
    session = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID, state=state)
    message = types.Content(role="user", parts=[types.Part(text="Write the brief.")])

    final_text = None
    async for event in runner.run_async(user_id=USER_ID, session_id=session.id, new_message=message):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    final_text = part.text

    assert final_text, "Analyst agent produced no final text response"
    return final_text


def _assert_baseline_constraints(brief_text: str):
    lowered = brief_text.lower()
    assert DISCLAIMER_PHRASE in lowered
    for phrase in FORBIDDEN_PHRASES:
        assert phrase not in lowered


@pytest.mark.asyncio
async def test_analyst_cites_seeded_numbers_not_hallucinated_ones():
    generated_at = datetime.now(timezone.utc).isoformat()
    research_data = {
        "success": True,
        "ticker": "ZZZZTEST",
        "price": 123.45,
        "pe_ratio": 67.89,
        "revenue_growth": 0.42,
    }
    sentiment_data = {
        "success": True,
        "ticker": "ZZZZTEST",
        "sentiment": "bullish",
        "cited_headlines": [
            {
                "headline": "Widgets Inc posts record quarter",
                "source": "TestWire",
                "url": "https://example.com/widgets",
                "published_at": generated_at,
            }
        ],
    }

    brief_text = await _run_analyst_with_seeded_state(
        {
            "ticker": "ZZZZTEST",
            "generated_at": generated_at,
            "research_data": json.dumps(research_data),
            "sentiment_data": json.dumps(sentiment_data),
        }
    )

    _assert_baseline_constraints(brief_text)
    assert _number_appears(brief_text, 123.45)
    assert _number_appears(brief_text, 67.89)
    assert "Widgets Inc posts record quarter" in brief_text
    assert generated_at in brief_text or generated_at[:10] in brief_text


@pytest.mark.asyncio
async def test_analyst_reports_unavailable_data_without_inventing_numbers():
    generated_at = datetime.now(timezone.utc).isoformat()
    research_data = {
        "success": False,
        "ticker": "ZZZZTEST",
        "error": "No price history found for ticker 'ZZZZTEST'. It may be invalid or delisted.",
    }
    sentiment_data = {
        "success": True,
        "ticker": "ZZZZTEST",
        "sentiment": "neutral",
        "cited_headlines": [],
    }

    brief_text = await _run_analyst_with_seeded_state(
        {
            "ticker": "ZZZZTEST",
            "generated_at": generated_at,
            "research_data": json.dumps(research_data),
            "sentiment_data": json.dumps(sentiment_data),
        }
    )

    _assert_baseline_constraints(brief_text)
    lowered = brief_text.lower()
    assert "unavailable" in lowered or "not available" in lowered


@pytest.mark.asyncio
async def test_end_to_end_aapl_brief_contains_real_data():
    direct_research = get_stock_data("AAPL")
    assert direct_research["success"] is True, "Precondition: direct tool call must succeed"

    brief = await run_research_brief("AAPL")

    assert brief.ticker == "AAPL"
    assert brief.research["success"] is True
    assert brief.sentiment["success"] is True

    lowered = brief.brief_markdown.lower()
    assert DISCLAIMER_PHRASE in lowered
    assert "data as of" in lowered
    assert "AAPL" in brief.brief_markdown

    # The brief's price must trace back to what the Research Agent's tool
    # actually returned in this same run, not a hallucinated figure.
    assert _number_appears(brief.brief_markdown, brief.research["price"])

    for phrase in FORBIDDEN_PHRASES:
        assert phrase not in lowered
