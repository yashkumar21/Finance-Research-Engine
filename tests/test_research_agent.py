"""End-to-end test of the Research Agent via ADK's InMemoryRunner.

Requires GOOGLE_API_KEY to be set (see .env.example). Makes real calls to
Gemini 2.5 Flash and to Yahoo Finance (through the agent's tool call).
"""

import json

import pytest
from dotenv import load_dotenv
from google.genai import types

from google.adk.runners import InMemoryRunner

from agents.research_agent import research_agent
from schemas import StockData, StockDataError
from tools.stock_data_tool import get_stock_data

load_dotenv()

APP_NAME = "finance_research_engine_test"
USER_ID = "test_user"


async def _run_agent(ticker: str) -> dict:
    runner = InMemoryRunner(agent=research_agent, app_name=APP_NAME)
    session = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID)

    message = types.Content(role="user", parts=[types.Part(text=f"Research {ticker}")])

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
async def test_research_agent_returns_valid_json_matching_tool_output():
    ticker = "AAPL"
    direct = get_stock_data(ticker)
    assert direct["success"] is True, "Precondition: direct tool call must succeed for this test"

    parsed = await _run_agent(ticker)

    data = StockData.model_validate(parsed)
    assert data.success is True
    assert data.ticker == ticker
    # The agent must pass through the tool's numbers, not invent its own.
    # rel=1e-2 tolerates live price/P-E ticking between the two independent
    # real-time calls (direct call vs. the agent's own tool call moments
    # later) - it's still tight enough to catch genuine hallucination.
    assert data.price == pytest.approx(direct["price"], rel=1e-2)
    if direct["pe_ratio"] is not None:
        assert data.pe_ratio == pytest.approx(direct["pe_ratio"], rel=1e-2)
    if direct["revenue_growth"] is not None:
        assert data.revenue_growth == pytest.approx(direct["revenue_growth"], rel=1e-6)


@pytest.mark.asyncio
async def test_research_agent_reports_structured_error_for_invalid_ticker():
    ticker = "ZZZZZZINVALID"
    parsed = await _run_agent(ticker)

    error = StockDataError.model_validate(parsed)
    assert error.success is False
    assert error.ticker == ticker
    assert error.error
