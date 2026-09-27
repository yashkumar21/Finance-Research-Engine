"""Research Agent: quantitative-only ticker research, wraps the Finnhub stock data tool."""

from google.adk.agents import Agent

from tools.stock_data_tool import get_stock_data

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
