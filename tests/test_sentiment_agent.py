"""End-to-end test of the Sentiment Agent via ADK's InMemoryRunner.

Requires GOOGLE_API_KEY and FINNHUB_API_KEY to be set. Makes real calls to
Gemini and to Finnhub (through the agent's tool call).
"""

import json

import pytest
from dotenv import load_dotenv
from google.genai import types

from google.adk.runners import InMemoryRunner

from agents.sentiment_agent import sentiment_agent
from schemas import SentimentAssessment, SentimentAssessmentError
# The agent's tool filters Finnhub's feed to articles about the company.
from tools.news_tool import get_relevant_company_news as get_company_news

load_dotenv()

APP_NAME = "finance_research_engine_test"
USER_ID = "test_user"


async def _run_agent(ticker: str) -> dict:
    runner = InMemoryRunner(agent=sentiment_agent, app_name=APP_NAME)
    session = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID)

    message = types.Content(role="user", parts=[types.Part(text=f"What's the news sentiment on {ticker}?")])

    final_text = None
    async for event in runner.run_async(user_id=USER_ID, session_id=session.id, new_message=message):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    final_text = part.text

    assert final_text, "Agent produced no final text response"

    cleaned = final_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    return json.loads(cleaned)


@pytest.mark.asyncio
async def test_sentiment_agent_cites_real_headlines_for_aapl():
    ticker = "AAPL"
    direct = get_company_news(ticker)
    assert direct["success"] is True, "Precondition: direct tool call must succeed for this test"

    parsed = await _run_agent(ticker)

    assessment = SentimentAssessment.model_validate(parsed)
    assert assessment.success is True
    assert assessment.ticker == ticker
    assert assessment.sentiment in {"bullish", "bearish", "neutral"}

    # Every cited headline must be a real one the tool actually returned -
    # the agent must not invent or paraphrase citations.
    real_headline_texts = {h["headline"] for h in direct["headlines"]}
    for cited in assessment.cited_headlines:
        assert cited.headline in real_headline_texts


@pytest.mark.asyncio
async def test_sentiment_agent_no_coverage_yields_neutral_with_no_citations():
    ticker = "UAMY"
    direct = get_company_news(ticker)
    assert direct["success"] is True

    parsed = await _run_agent(ticker)
    assessment = SentimentAssessment.model_validate(parsed)
    assert assessment.success is True

    if not direct["headlines"]:
        assert assessment.sentiment == "neutral"
        assert assessment.cited_headlines == []


@pytest.mark.asyncio
async def test_sentiment_agent_reports_structured_error_for_invalid_ticker():
    ticker = "ZZZZZZINVALID"
    parsed = await _run_agent(ticker)

    error = SentimentAssessmentError.model_validate(parsed)
    assert error.success is False
    assert error.ticker == ticker
    assert error.error
