"""Research Agent: quantitative-only ticker research, wraps the Finnhub stock data tool.

Two versions:
- build_research_step(): what the brief pipeline uses - a plain Python step
  that calls get_stock_data and puts its result in state. A model adds
  nothing here (it only called the tool and copied the numbers), and its
  Gemini calls were where briefs stalled.
- build_research_agent(): the original LLM agent, kept for
  run_research_agent.py and its tests.
"""

import asyncio
import json
from typing import AsyncGenerator

from google.adk.agents import Agent, BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from tools.stock_data_tool import get_stock_data


class ResearchStep(BaseAgent):
    """Fetches the ticker's quantitative data with no model call."""

    output_key: str = "research_data"

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        data = json.dumps(await asyncio.to_thread(get_stock_data, ctx.session.state["ticker"]))
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            content=types.Content(role="model", parts=[types.Part(text=data)]),
            actions=EventActions(state_delta={self.output_key: data}),
        )


def build_research_step(output_key: str = "research_data") -> ResearchStep:
    """The brief pipeline's research step. Named research_agent so progress
    displays keyed on that author keep working."""
    return ResearchStep(name="research_agent", output_key=output_key)

RESEARCH_AGENT_INSTRUCTION = """\
You are a quantitative research agent. You report numbers only - you never
give buy/sell recommendations, price targets, or any investment judgment.

For every ticker you are asked about, call the get_stock_data tool exactly
once to fetch price, today's move, P/E ratio, revenue growth, 52-week range
and market cap.

After the tool returns, respond with ONLY a single JSON object and nothing
else - no markdown fences, no commentary, no explanation. The JSON object
must have exactly these keys, copied faithfully from the tool's output
(do not recompute or alter any values):

On success:
{
  "success": true,
  "ticker": "<string>",
  "price": <number or null>,
  "pe_ratio": <number or null>,
  "revenue_growth": <number or null>,
  "pct_change": <number or null>,
  "week52_high": <number or null>,
  "week52_low": <number or null>,
  "market_cap": <number or null>
}

On failure (the tool reported an error):
{
  "success": false,
  "ticker": "<string>",
  "error": "<string>"
}
"""

def build_research_agent(output_key: str | None = None) -> Agent:
    """Construct a fresh Research Agent instance.

    A factory rather than a shared singleton: ADK raises a ValidationError
    if the same agent object is attached as a sub-agent to more than one
    parent (e.g. reused inside orchestrator.py's ParallelAgent while also
    being the module-level singleton below), so composed pipelines must
    build their own instance via this function rather than importing
    research_agent directly.
    """
    return Agent(
        name="research_agent",
        model="gemini-3.5-flash-lite",
        description="Quantitative stock research agent: price, move, P/E, revenue growth, 52-week range, market cap.",
        instruction=RESEARCH_AGENT_INSTRUCTION,
        tools=[get_stock_data],
        output_key=output_key,
    )


research_agent = build_research_agent()

root_agent = research_agent
