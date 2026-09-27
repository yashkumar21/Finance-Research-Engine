"""Orchestrator (Workflow Router): runs the Research and Sentiment Agents in
parallel, then feeds both into the Analyst Agent for synthesis.

Uses ADK's ParallelAgent + SequentialAgent. Both are marked deprecated in
the installed adk-python version (2.5.0) in favor of a lower-level
Workflow/Node/Edge graph API, but that replacement's own docstring states
it "cannot yet be used as an LlmAgent sub-agent" and has no higher-level
convenience equivalent yet - so ParallelAgent/SequentialAgent remain the
current idiomatic, documented way to compose this exact fan-out/fan-in
pattern. Revisit if/when Workflow matures.
"""

import json
from datetime import datetime, timezone
from typing import Callable, Optional

from google.adk.agents import ParallelAgent, SequentialAgent
from google.adk.events import Event
from google.adk.runners import InMemoryRunner
from google.genai import types

from agents.analyst_agent import build_analyst_agent
from agents.figures import (
    build_key_figures_agent,
    correct_citation_dates,
    format_as_of,
    insert_key_figures,
)
from agents.research_agent import build_research_agent
from agents.sentiment_agent import build_sentiment_agent
from pricing import gemini_cost
from schemas import ResearchBrief, Usage
from tools.stock_data_tool import resolve_ticker

APP_NAME = "finance_research_engine_orchestrator"


def build_pipeline() -> SequentialAgent:
    """Build a fresh Research || Sentiment -> key figures -> Analyst pipeline.

    The key-figures step is plain Python, not a model: it formats the
    Research Agent's numbers so the Analyst only ever quotes them.

    Fresh agent instances every call (via the build_* factories) - ADK
    forbids attaching the same agent object as a sub-agent of more than one
    parent, so this must never reuse the module-level singletons.
    """
    research_step = build_research_agent(output_key="research_data")
    sentiment_step = build_sentiment_agent(output_key="sentiment_data")

    parallel_step = ParallelAgent(
        name="research_and_sentiment_parallel",
        sub_agents=[research_step, sentiment_step],
    )

    analyst_step = build_analyst_agent()

    return SequentialAgent(
        name="finance_research_pipeline",
        sub_agents=[parallel_step, build_key_figures_agent(), analyst_step],
    )


def _parse_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    return json.loads(cleaned)


async def run_research_brief(
    ticker: str,
    user_id: str = "orchestrator_user",
    on_event: Optional[Callable[[Event], None]] = None,
) -> ResearchBrief:
    """Run the full pipeline for a ticker and return the assembled brief.

    on_event, if given, is called synchronously once per streamed event
    (e.g. to drive live per-agent status in a UI) - purely observational,
    it cannot alter pipeline execution.

    The brief's usage sums token counts across all three agents; its cost is
    an estimate (tokens x list price, see pricing.py).
    """
    ticker = resolve_ticker(ticker)
    pipeline = build_pipeline()
    runner = InMemoryRunner(agent=pipeline, app_name=APP_NAME)

    generated_at = datetime.now(timezone.utc).isoformat()
    session = await runner.session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        state={"ticker": ticker, "generated_at": generated_at, "as_of": format_as_of(generated_at)},
    )

    message = types.Content(role="user", parts=[types.Part(text=f"Research {ticker}")])

    final_text = None
    input_tokens = output_tokens = 0
    async for event in runner.run_async(user_id=user_id, session_id=session.id, new_message=message):
        if on_event:
            on_event(event)
        usage = event.usage_metadata
        if usage:
            input_tokens += usage.prompt_token_count or 0
            # Thinking tokens are billed as output.
            output_tokens += (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    final_text = part.text

    assert final_text, "Analyst agent produced no final markdown response"

    final_session = await runner.session_service.get_session(
        app_name=APP_NAME, user_id=user_id, session_id=session.id
    )
    state = final_session.state

    research_data = _parse_json(state["research_data"])
    sentiment_data = _parse_json(state["sentiment_data"])

    return ResearchBrief(
        ticker=ticker,
        research=research_data,
        sentiment=sentiment_data,
        brief_markdown=correct_citation_dates(insert_key_figures(final_text, state["key_figures"]), sentiment_data),
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=gemini_cost(input_tokens, output_tokens),
            cost_is_estimate=True,
        ),
    )
