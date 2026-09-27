"""Sentiment Agent: qualitative-only news sentiment, wraps the Finnhub news tool."""

from google.adk.agents import Agent

from tools.news_tool import get_relevant_company_news


def get_company_news(ticker: str) -> dict:
    """Fetch recent news headlines about a company, by stock ticker.

    Returns only headlines that mention the company by name or ticker, from
    the last 7 days - raw headlines, sources and URLs, with no sentiment
    analysis. An empty list with a note means no recent coverage of the
    company itself.

    Args:
        ticker: The stock ticker symbol to look up, e.g. "AAPL", "MSFT".
    """
    # Named get_company_news because the instruction refers to the tool by
    # that name; filters Finnhub's feed, which mixes in other companies' news.
    return get_relevant_company_news(ticker)


SENTIMENT_AGENT_INSTRUCTION = """\
You are a qualitative sentiment research agent. You never give price
targets, valuations, or buy/sell recommendations - you only assess the tone
of recent news coverage.

For every ticker you are asked about, call the get_company_news tool exactly
once to fetch recent headlines.

The tool returns raw headlines only - it does NOT tell you the sentiment.
You must read the headlines yourself and judge the overall sentiment:
"bullish" if the headlines skew positive for the company, "bearish" if they
skew negative, or "neutral" if they are mixed, non-committal, or there is no
recent coverage. Reason over the headline text itself; do not rely on any
external sentiment score.

After the tool returns, respond with ONLY a single JSON object and nothing
else - no markdown fences, no commentary, no explanation.

On success:
{
  "success": true,
  "ticker": "<string>",
  "sentiment": "bullish" | "bearish" | "neutral",
  "cited_headlines": [
    {
      "headline": "<string, copied exactly from the tool output>",
      "source": "<string>",
      "url": "<string>",
      "published_at": "<string>"
    }
  ]
}

cited_headlines must be a subset of the headlines the tool returned that you
actually relied on to reach your sentiment judgment - copy each field
faithfully from the tool's output, never paraphrase or invent a headline. If
the tool returned no headlines, use sentiment "neutral" and an empty
cited_headlines list.

On failure (the tool reported an error):
{
  "success": false,
  "ticker": "<string>",
  "error": "<string>"
}
"""

def build_sentiment_agent(output_key: str | None = None, news_tool=get_company_news) -> Agent:
    """Construct a fresh Sentiment Agent instance.

    A factory rather than a shared singleton - see build_research_agent's
    docstring for why: ADK forbids attaching the same agent object as a
    sub-agent to more than one parent.

    news_tool lets the eval harness substitute a function that returns
    cached headlines, so Gemini and Jev judge identical inputs. It must keep
    the name get_company_news, since the instruction refers to it by name.
    """
    return Agent(
        name="sentiment_agent",
        model="gemini-3.5-flash-lite",
        description="Qualitative news-sentiment agent: bullish/bearish/neutral judgment from recent headlines, with citations.",
        instruction=SENTIMENT_AGENT_INSTRUCTION,
        tools=[news_tool],
        output_key=output_key,
    )


sentiment_agent = build_sentiment_agent()

root_agent = sentiment_agent
