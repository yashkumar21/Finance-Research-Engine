# Finance Research Engine

Multi-agent financial research system built with Google's Agent Development Kit (ADK) and Gemini.
The Research Agent currently uses `gemini-3.6-flash` (originally targeted `gemini-2.5-flash`,
which was deprecated for new API keys — see `agents/research_agent.py`).

Full target architecture: a Workflow Router triggers a Research Agent (quantitative)
and Sentiment Agent (qualitative) in parallel, then an Analyst Agent synthesizes
both into a markdown research brief. A Streamlit frontend will sit on top later.

**Currently implemented**: the yfinance tool and the Research Agent only.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your GOOGLE_API_KEY from https://aistudio.google.com/apikey
```

## Project layout

- `schemas.py` — shared Pydantic models (`StockData`, `StockDataError`) for the tool's output shape.
- `tools/stock_data_tool.py` — `get_stock_data(ticker)`, wraps yfinance, never raises.
- `agents/research_agent.py` — ADK `Agent` wrapping the tool, quantitative-only, JSON-only output.
- `run_research_agent.py` — manual CLI: `python run_research_agent.py AAPL`
- `tests/` — pytest suite (real network calls, no mocking).

## Running tests

```bash
# Tool tests only (no API key needed)
pytest tests/test_stock_data_tool.py -v

# Full suite, including the agent (needs GOOGLE_API_KEY in .env)
pytest -v
```

## Manual run

```bash
python run_research_agent.py AAPL
python run_research_agent.py ZZZZZZINVALID
```
