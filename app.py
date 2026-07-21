"""Streamlit frontend for the Finance Research Engine."""

import os
from pathlib import Path

import streamlit as st

# Must be the very first Streamlit command in the script - touching
# st.secrets (in _resolve_secret below) before this renders output of its
# own when no secrets.toml exists, which then makes set_page_config raise.
st.set_page_config(page_title="Finance Research Engine", page_icon="📈")

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

# st.secrets itself writes a warning to the page the moment it's touched if
# neither of these files exists - not just a catchable Python exception -
# so check for the file first and only touch st.secrets when one is there.
_SECRETS_PATHS = [
    Path.home() / ".streamlit" / "1.toml",
    Path(__file__).parent / ".streamlit" / "secrets.toml",
]
_SECRETS_AVAILABLE = any(p.exists() for p in _SECRETS_PATHS)


def _resolve_secret(key: str):
    """st.secrets when deployed (Streamlit Community Cloud), .env locally."""
    if _SECRETS_AVAILABLE:
        try:
            if key in st.secrets:
                return st.secrets[key]
        except Exception:
            pass
    return os.environ.get(key)


# Must happen before importing agents.orchestrator, so the tools/agents
# (which read these via os.environ / their own load_dotenv()) see them.
for _key in ("GOOGLE_API_KEY", "FINNHUB_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI"):
    _value = _resolve_secret(_key)
    if _value:
        os.environ[_key] = _value

import asyncio  # noqa: E402

from agents.orchestrator import run_research_brief  # noqa: E402

st.title("📈 Finance Research Engine")
st.caption(
    "Multi-agent research: quantitative data + news sentiment, run in parallel "
    "and synthesized into a brief."
)

missing_keys = [k for k in ("GOOGLE_API_KEY", "FINNHUB_API_KEY") if not os.environ.get(k)]
if missing_keys:
    st.warning(
        f"Missing configuration: {', '.join(missing_keys)}. "
        "Set these in your local .env file, or in Streamlit secrets when deployed."
    )

with st.form("ticker_form"):
    ticker_input = st.text_input("Stock ticker", placeholder="AAPL")
    submitted = st.form_submit_button("Research")

if submitted:
    ticker = (ticker_input or "").strip()
    if not ticker:
        st.warning("Please enter a ticker symbol.")
    else:
        progress = {"research_done": False, "sentiment_done": False, "analyst_started": False}
        brief = None

        with st.status("Fetching quantitative + sentiment data...", expanded=True) as status:

            def on_event(event):
                has_text = bool(
                    event.content and event.content.parts and any(p.text for p in event.content.parts)
                )
                if not has_text:
                    return
                if event.author == "research_agent" and not progress["research_done"]:
                    progress["research_done"] = True
                    status.write("✅ Research Agent: quantitative data received")
                elif event.author == "sentiment_agent" and not progress["sentiment_done"]:
                    progress["sentiment_done"] = True
                    status.write("✅ Sentiment Agent: news sentiment received")
                elif event.author == "analyst_agent" and not progress["analyst_started"]:
                    progress["analyst_started"] = True
                    status.update(label="Synthesizing research brief...")
                    status.write("✍️ Analyst Agent: synthesizing brief")

            try:
                brief = asyncio.run(run_research_brief(ticker, on_event=on_event))
            except Exception as exc:
                status.update(label="Failed", state="error")
                st.error(
                    "Something went wrong while generating the research brief. "
                    "This is usually a missing/invalid API key or a temporary rate limit. "
                    f"Details: {exc}"
                )
            else:
                status.update(label="Research brief ready", state="complete")

        if brief is not None:
            if not brief.research.get("success", True):
                st.warning(f"Quantitative data unavailable for {brief.ticker}: {brief.research.get('error')}")
            if not brief.sentiment.get("success", True):
                st.warning(f"Sentiment data unavailable for {brief.ticker}: {brief.sentiment.get('error')}")

            st.markdown(brief.brief_markdown)

            with st.expander("Research Agent output (raw)"):
                st.json(brief.research)

            with st.expander("Sentiment Agent output (raw)"):
                st.json(brief.sentiment)
