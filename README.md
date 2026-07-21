# Finance Research Engine

Multi-agent financial research system built with Google's Agent Development Kit (ADK) and Gemini.
Both agents currently use `gemini-3.5-flash-lite`. This has moved twice already: the original
target `gemini-2.5-flash` was deprecated for new API keys, its replacement `gemini-3.6-flash`
hit its free-tier daily quota (20 requests/day/model) from repeated test runs, and
`gemini-2.5-flash-lite` turned out to be deprecated too. See `agents/research_agent.py` and
`agents/sentiment_agent.py` — swap the `model=` string there if you hit another quota wall or
want a different model.

Full target architecture: a Workflow Router triggers a Research Agent (quantitative)
and Sentiment Agent (qualitative) in parallel, then an Analyst Agent synthesizes
both into a markdown research brief. A Streamlit frontend will sit on top later.

**Currently implemented**: the yfinance tool + Research Agent, and the Finnhub news tool + Sentiment Agent.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in GOOGLE_API_KEY (https://aistudio.google.com/apikey)
                        # and FINNHUB_API_KEY (https://finnhub.io/register, free tier, 60 calls/min)
```

## Project layout

- `schemas.py` — shared Pydantic models for both tools' output shapes: `StockData`/`StockDataError`,
  `Headline`, `NewsData`/`NewsDataError`, `SentimentAssessment`/`SentimentAssessmentError`.
- `tools/stock_data_tool.py` — `get_stock_data(ticker)`, wraps yfinance, never raises.
- `tools/news_tool.py` — `get_company_news(ticker)`, wraps Finnhub, headlines only (no sentiment
  computed here), never raises.
- `agents/research_agent.py` — ADK `Agent` wrapping the stock data tool, quantitative-only, JSON-only output.
- `agents/sentiment_agent.py` — ADK `Agent` wrapping the news tool; the agent itself does the
  bullish/bearish/neutral judgment by reading the raw headlines, and must cite the specific
  headlines (copied verbatim from the tool) it based the call on.
- `run_research_agent.py` — manual CLI: `python run_research_agent.py AAPL`
- `tests/` — pytest suite. Real network calls, no mocking, with one exception: Finnhub
  rate-limit/network-failure handling is tested via `monkeypatch` (deliberately triggering a
  real 429 would be slow, flaky, and burn shared free-tier quota).

## Running tests

```bash
# Tool tests only (no LLM calls, no GOOGLE_API_KEY needed)
pytest tests/test_stock_data_tool.py tests/test_news_tool.py -v

# Full suite, including both agents (needs GOOGLE_API_KEY and FINNHUB_API_KEY in .env)
pytest -v
```

## Manual run

```bash
python run_research_agent.py AAPL
python run_research_agent.py ZZZZZZINVALID
```
