"""Research Agent: quantitative-only ticker research, wraps the yfinance tool."""

from google.adk.agents import Agent

from tools.stock_data_tool import get_stock_data

RESEARCH_AGENT_INSTRUCTION = """\
You are a quantitative research agent. You report numbers only - you never
give buy/sell recommendations, price targets, or any investment judgment.

For every ticker you are asked about, call the get_stock_data tool exactly
once to fetch price, P/E ratio, and revenue growth.

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
  "revenue_growth": <number or null>
}

On failure (the tool reported an error):
{
  "success": false,
  "ticker": "<string>",
  "error": "<string>"
}
"""

research_agent = Agent(
    name="research_agent",
    model="gemini-3.5-flash-lite",
    description="Quantitative stock research agent: price, P/E ratio, and revenue growth only.",
    instruction=RESEARCH_AGENT_INSTRUCTION,
    tools=[get_stock_data],
)

root_agent = research_agent
