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
    Path.home() / ".streamlit" / "secrets.toml",
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
for _key in ("GOOGLE_API_KEY", "FINNHUB_API_KEY", "TYPESAFE_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI"):
    _value = _resolve_secret(_key)
    if _value:
        os.environ[_key] = _value

import asyncio  # noqa: E402
import statistics  # noqa: E402

import pandas as pd  # noqa: E402

from agents.orchestrator import run_research_brief  # noqa: E402
from schemas import ScanReport  # noqa: E402
from screener.scan import list_reports, run_scan, save_report  # noqa: E402

LIVE_SCAN_MAX_TICKERS = 20
LIVE_SCAN_MAX_BRIEFS = 3

st.title("📈 Finance Research Engine")
st.caption(
    "Multi-agent research: quantitative data + news sentiment, run in parallel "
    "and synthesized into a brief. A cheap Jev screen decides which tickers in a "
    "nightly scan deserve one."
)



@st.cache_data
def _load_report(path: str, mtime: float) -> ScanReport:
    """Parsed report; mtime is part of the cache key so rewrites reload."""
    return ScanReport.model_validate_json(Path(path).read_text())


def _mean_brief_cost(reports: list[Path]) -> tuple[float, str] | None:
    """Mean Gemini brief cost from the newest report that ran briefs."""
    for path in reports:
        report = _load_report(str(path), path.stat().st_mtime)
        costs = [r.brief.usage.cost_usd for r in report.results if r.brief and r.brief.usage]
        if costs:
            return statistics.mean(costs), path.name
    return None


def _report_label(path: Path) -> str:
    report = _load_report(str(path), path.stat().st_mtime)
    mode = f"{report.briefs_generated} briefs" if report.escalation_enabled else "Jev only"
    return f"{report.started_at[:16].replace('T', ' ')} UTC - {report.universe}, {report.tickers_scanned} tickers ({mode})"


def _usd(amount: float) -> str:
    """Enough decimals that a few Jev calls don't round to $0.0000."""
    return f"${amount:.4f}" if amount >= 0.001 else f"${amount:.6f}"


def _results_table(report: ScanReport) -> pd.DataFrame:
    rows = []
    for r in report.results:
        s = r.screen
        rows.append({
            "Ticker": r.ticker,
            "Move %": r.pct_change,
            "Escalate": "yes" if r.decision.escalate else "no",
            "Sentiment": s.sentiment if s else "error",
            "Confidence": s.sentiment_confidence if s else None,
            "Material event": s.material_event if s else None,
            "Needs analysis": s.needs_analysis if s else None,
            "Headlines": r.headline_count,
            "Reasons": "; ".join(r.decision.reasons) or ("" if s else r.screen_error),
            "Brief": "yes" if r.brief else ("failed" if r.brief_error else ""),
        })
    return pd.DataFrame(rows)


def _render_report(report: ScanReport, reports: list[Path]) -> None:
    cols = st.columns(4)
    cols[0].metric("Tickers scanned", report.tickers_scanned)
    cols[1].metric("Escalated", f"{report.escalation_rate:.0%}", help=f"{report.tickers_escalated} tickers")
    cols[2].metric(
        f"Jev cost ({'estimated' if report.jev_cost_is_estimate else 'exact'})", _usd(report.jev_cost_usd),
        help=f"Median latency {report.jev_latency_p50_ms} ms, p95 {report.jev_latency_p95_ms} ms",
    )

    # Savings vs. baseline A (a full Gemini brief for every ticker), projected
    # at a measured per-brief cost - an estimate, like all Gemini costs here.
    brief_cost = _mean_brief_cost(reports)
    if brief_cost:
        per_brief, source = brief_cost
        baseline = per_brief * report.tickers_scanned
        screened = report.jev_cost_usd + per_brief * report.tickers_escalated
        cols[3].metric(
            "Projected savings", f"{1 - screened / baseline:.0%}",
            help=(
                f"vs. briefing every ticker: ${screened:.3f} (Jev + briefs for escalated tickers) "
                f"instead of ${baseline:.3f}. Projected at ${per_brief:.4f}/brief, the mean "
                f"estimated Gemini cost in {source}."
            ),
        )
    else:
        cols[3].metric("Projected savings", "n/a", help="Needs one scan that ran briefs to measure brief cost.")

    if not report.escalation_enabled:
        st.info(
            "Jev-only scan: every decision is recorded but no briefs ran. "
            f"{report.tickers_escalated} of {report.tickers_scanned} tickers would have been escalated."
        )
    elif report.max_briefs is not None and report.tickers_escalated > report.briefs_generated:
        st.info(
            f"Briefs were capped at {report.max_briefs}: the strongest signals were briefed, "
            f"{report.tickers_escalated - report.briefs_generated} escalated tickers were not."
        )

    show = st.segmented_control(
        "Show", ["All", "Escalated", "Not escalated"], default="All", key="scan_filter"
    ) or "All"
    table = _results_table(report)
    if show != "All":
        table = table[table["Escalate"] == ("yes" if show == "Escalated" else "no")]

    probability = dict(min_value=0.0, max_value=1.0, format="%.2f")
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "Move %": st.column_config.NumberColumn(format="%+.2f"),
            "Confidence": st.column_config.ProgressColumn(**probability, help="Jev's confidence in its sentiment call"),
            "Material event": st.column_config.ProgressColumn(**probability, help="P(headlines report a material event)"),
            "Needs analysis": st.column_config.ProgressColumn(**probability, help="P(a covering analyst would update their view)"),
            "Reasons": st.column_config.TextColumn(width="large"),
        },
    )

    briefed = [r for r in report.results if r.brief or r.brief_error]
    if briefed:
        st.subheader("Briefs for escalated tickers")
        for r in briefed:
            with st.expander(f"{r.ticker} - {'; '.join(r.decision.reasons)}"):
                if r.brief:
                    st.markdown(r.brief.brief_markdown)
                    if r.brief.usage:
                        st.caption(f"Estimated Gemini cost: ${r.brief.usage.cost_usd:.4f}")
                else:
                    st.error(f"Brief failed: {r.brief_error}")


def _render_live_scan() -> None:
    with st.expander(f"Run a small live scan (up to {LIVE_SCAN_MAX_TICKERS} tickers)"):
        with st.form("live_scan_form"):
            tickers_input = st.text_input("Tickers, comma-separated", placeholder="AAPL, MSFT, NVDA")
            with_briefs = st.checkbox(f"Also brief escalated tickers (up to {LIVE_SCAN_MAX_BRIEFS}, uses Gemini quota)")
            submitted = st.form_submit_button("Scan")
        if not submitted:
            return
        tickers = list(dict.fromkeys(t.strip().upper() for t in tickers_input.split(",") if t.strip()))
        if not tickers:
            st.warning("Enter at least one ticker.")
            return
        if len(tickers) > LIVE_SCAN_MAX_TICKERS:
            st.warning(f"Live scans are capped at {LIVE_SCAN_MAX_TICKERS} tickers - use run_scan.py for more.")
            return
        with st.status(f"Scanning {len(tickers)} tickers...", expanded=True) as status:
            def on_result(done, total, result):
                flag = "escalate" if result.decision.escalate else "skip"
                status.write(f"{done}/{total} {result.ticker}: {flag}")

            try:
                report = asyncio.run(run_scan(
                    tickers, "live", escalate=with_briefs,
                    max_briefs=LIVE_SCAN_MAX_BRIEFS, on_result=on_result,
                ))
            except Exception as exc:
                status.update(label="Scan failed", state="error")
                st.error(f"Scan failed: {exc}")
                return
            save_report(report)
            status.update(label="Scan complete - showing it below", state="complete")


tab_brief, tab_scan = st.tabs(["Single ticker", "Daily Scan"])

with tab_brief:
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


with tab_scan:
    missing_scan_keys = [k for k in ("TYPESAFE_API_KEY", "FINNHUB_API_KEY") if not os.environ.get(k)]
    if missing_scan_keys:
        st.warning(f"Live scans need {', '.join(missing_scan_keys)}.")
    _render_live_scan()

    reports = list_reports()
    if not reports:
        st.info("No scans yet. Run `python run_scan.py --universe sp100 --no-escalate` to create one.")
    else:
        selected = st.selectbox("Scan report", reports, format_func=_report_label)
        _render_report(_load_report(str(selected), selected.stat().st_mtime), reports)
