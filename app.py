"""Streamlit frontend for the Finance Research Engine."""

import os
from pathlib import Path

import streamlit as st

# Must be the very first Streamlit command in the script - touching
# st.secrets (in _resolve_secret below) before this renders output of its
# own when no secrets.toml exists, which then makes set_page_config raise.
st.set_page_config(page_title="Finance Research Engine", page_icon="📈", layout="wide")

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
from functools import partial  # noqa: E402

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402

import re  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

from agents.orchestrator import (  # noqa: E402
    BRIEF_ATTEMPTS,
    BRIEF_TIMEOUT_SECONDS,
    BriefTimeoutError,
    run_research_brief,
)
from schemas import ScanReport  # noqa: E402
from screener.policy import DEFAULT_POLICY, decide  # noqa: E402
from screener.scan import is_demo_report, list_reports, load_company_names, run_scan, save_report  # noqa: E402
from tools.jev_screen import screen_ticker  # noqa: E402
from tools.news_tool import get_company_news, headlines_about  # noqa: E402
from tools.stock_data_tool import get_quote_snapshot, resolve_ticker  # noqa: E402

LIVE_SCAN_MAX_TICKERS = 20
LIVE_SCAN_MAX_BRIEFS = 3
# Typical time of a full brief (9-12 s since research became a Python step), for the button label.
# Per-click costs stay out of the single-ticker flow; the Daily Scan tab shows costs.
BRIEF_TIME_HINT = "~10 s"
# One-click examples so a first-time visitor doesn't have to think of a ticker.
EXAMPLE_TICKERS = {"AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia", "TSLA": "Tesla"}

# Held-out evaluation of the current policy, from the committed reports in
# eval/results/ (the JSON results are gitignored, so a deployed app can't
# recompute these). Update alongside the README's Results section.
HELD_OUT = {
    "tickers": 100,
    "events": 8,
    "caught": 8,
    "escalation": "26-30%",
    "precision": "27-31%",
    "runs": 5,
    "report": "eval/results/compare-2026-09-27T203931Z.md",
}

METHODOLOGY_MD = f"""
**How a nightly scan works**

1. For each S&P 500 ticker, fetch today's price move and the last 7 days of headlines (Finnhub).
2. **Jev** (TypeSafe's "System 1" decision model) answers typed questions about those headlines in one
   cheap call (~$0.00004): overall sentiment, and the probability that a *material event* happened.
3. A deterministic policy escalates a ticker when **material event >= 0.6**, the price moved **>= 4%**,
   or screening failed (never silently dropped).
4. Escalated tickers get the full Gemini research brief (research + sentiment agents in parallel,
   then an analyst agent); the rest are recorded without one.

**How it was evaluated**

- *Material event* has a written definition: an earnings surprise, guidance change, M&A, major
  litigation, regulatory action, or executive change (`eval/LABELING.md`).
- Tickers were hand-labeled blind (headlines only, never a model's answer).
- The policy was chosen on 102 labeled S&P 100 tickers (7 events, all caught at 26% escalation), then
  **committed before** a held-out set was labeled.
- **Held-out test ({HELD_OUT['tickers']} random S&P 500 tickers outside the S&P 100):** caught
  **{HELD_OUT['caught']} of {HELD_OUT['events']}** material events at **{HELD_OUT['escalation']}**
  escalation, across {HELD_OUT['runs']} runs and three Jev providers ({HELD_OUT['report']}).

**Limitations**

- Only 15 labeled events so far; 8/8 is consistent with a true recall as low as ~63%.
- About 70% of escalations are false alarms - acceptable for a screen that must not miss events,
  but there is room to cut briefs further.
- Headlines are the only input; Gemini costs, and Jev's on TypeSafe's API, are estimates from tokens.

*A research aid, not investment advice.*
"""

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


def _report_label(path: Path) -> str:
    report = _load_report(str(path), path.stat().st_mtime)
    when = datetime.fromisoformat(report.started_at).strftime("%-d %b %Y, %H:%M UTC")
    universe = {"sp500": "S&P 500", "sp100": "S&P 100", "live": "Live scan", "custom": "Custom list"}.get(
        report.universe, report.universe
    )
    n = report.briefs_generated
    mode = f"{n} brief{'s' if n != 1 else ''}" if report.escalation_enabled else "screening only"
    return f"{when} · {universe} · {report.tickers_scanned} tickers · {mode}"


def _md(text: str) -> str:
    """Markdown-safe text: Streamlit renders $...$ as LaTeX, and headlines and
    briefs are full of dollar amounts ("$10 Billion ... $38 Billion")."""
    return text.replace("$", "\\$")


def _brief_md(markdown: str) -> str:
    """A brief for display inside the page: headings two levels smaller (its
    "# Title" would otherwise match the app's own title) and $ escaped."""
    smaller = re.sub(r"^(#{1,4}) ", lambda m: "#" * (len(m[1]) + 2) + " ", markdown, flags=re.MULTILINE)
    return _md(smaller)


_BOTTOM_LINE = re.compile(r"^## Bottom line[ \t]*\n(.*?)(?=^## |\Z)", re.MULTILINE | re.DOTALL)


def _compact_brief(markdown: str) -> None:
    """A brief for the pop-up: the Bottom line up front, the rest (key figures,
    news, things to watch, sources) behind an expander so the pop-up stays short.
    Falls back to the whole brief if it has no Bottom line section."""
    match = _BOTTOM_LINE.search(markdown)
    if not match:
        with st.container(border=True):
            st.markdown(_brief_md(markdown))
        return
    as_of = re.search(r"^\*Data as of .*\*$", markdown, re.MULTILINE)
    with st.container(border=True):
        st.markdown("**Research brief: bottom line**" + (f" · {as_of[0]}" if as_of else ""))
        st.markdown(_md(match[1].strip()))
        with st.expander("Read the full brief - key figures, news, what to watch, sources"):
            st.markdown(_brief_md(markdown[match.end():].strip()))


def _usd(amount: float) -> str:
    """Enough decimals that a few Jev calls don't round to $0.0000."""
    return f"${amount:.4f}" if amount >= 0.001 else f"${amount:.6f}"


@st.cache_data
def _company_names() -> dict[str, str]:
    return load_company_names()


_REASON_PATTERNS = [
    (re.compile(r"material event likely \(p=([\d.]+)\)"), lambda m: f"Major news likely ({float(m[1]):.0%})"),
    (re.compile(r"price moved ([+-][\d.]+)% today"), lambda m: f"Price moved {m[1]}% today"),
    (re.compile(r"analyst attention likely needed \(p=([\d.]+)\)"), lambda m: f"Analyst attention likely ({float(m[1]):.0%})"),
    (re.compile(r"Jev unsure of sentiment.*"), lambda m: "Unclear news tone"),
    (re.compile(r"(screening|scan) failed.*"), lambda m: "Screening failed - flagged to be safe"),
]


def _plain_reason(reason: str) -> str:
    """Policy reasons ("material event likely (p=0.98)") in reader-facing words.
    Also covers reports saved under the earlier policy's reasons."""
    for pattern, fmt in _REASON_PATTERNS:
        match = pattern.fullmatch(reason)
        if match:
            return fmt(match)
    return reason


def _why(r) -> str:
    """The table's Why column: flag reasons, a close call, or a dash."""
    if r.decision.escalate:
        return "; ".join(_plain_reason(x) for x in r.decision.reasons)
    if r.screen and _is_close_call(r.screen.material_event, False):
        return f"Close call ({r.screen.material_event:.0%})"
    return "–"


def _results_table(report: ScanReport) -> pd.DataFrame:
    """One row per ticker, flagged first and strongest signal first - what a
    reviewer looks at. needs_analysis and sentiment confidence are left out."""
    names = _company_names()
    rows = []
    for r in report.results:
        s = r.screen
        rows.append({
            "Ticker": r.ticker,
            "Company": names.get(r.ticker) or r.company_name or "",
            "Flagged": "yes" if r.decision.escalate else "no",
            "Why": _why(r),
            "Major-news likelihood": s.material_event if s else None,
            "Move %": round(r.pct_change, 2) + 0.0 if r.pct_change is not None else None,  # no "-0.00"
            "News tone": s.sentiment if s else "n/a",
            "Headlines": r.headline_count,
            "Brief": "yes" if r.brief else ("failed" if r.brief_error else ""),
        })
    table = pd.DataFrame(rows)
    return table.sort_values(
        ["Flagged", "Major-news likelihood"], ascending=[False, False], na_position="first", kind="stable"
    )


def _report_summary(report: ScanReport) -> str:
    """One line for the selected report - the top strip already has the headline numbers."""
    when = datetime.fromisoformat(report.started_at).strftime("%-d %b %Y, %H:%M UTC")
    universe = {"sp500": "S&P 500", "sp100": "S&P 100"}.get(report.universe, report.universe)
    if not report.escalation_enabled:
        mode = "screening only, no briefs run"
    elif report.max_briefs is not None and report.tickers_escalated > report.briefs_generated:
        mode = f"{report.briefs_generated} briefs, capped at {report.max_briefs}"
    else:
        mode = f"{report.briefs_generated} briefs"
    cost = f"{_usd(report.jev_cost_usd)} {'estimated' if report.jev_cost_is_estimate else 'exact'}"
    return (
        f"**{when}** · {universe} · {mode} · {report.tickers_scanned} screened, "
        f"**{report.tickers_escalated} flagged** ({report.escalation_rate:.0%}) · screening cost {cost}"
    )


# Flagged vs. not flagged: one accent against a recessive grey (checked for
# colour-blind separation in light and dark themes).
FLAGGED_COLOR = "#3987e5"
NOT_FLAGGED_COLOR = "#8b8a85"
# Reasons only the earlier policy (needs_analysis / low confidence) produced.
_EARLIER_POLICY_REASONS = ("analyst attention likely", "Jev unsure of sentiment")


def _scan_chart(table: pd.DataFrame, report: ScanReport) -> alt.LayerChart:
    """Major-news likelihood vs. today's move for every screened ticker, with the
    policy's escalation zones shaded - the whole screen in one picture."""
    data = table.dropna(subset=["Major-news likelihood", "Move %"]).rename(
        columns={"Major-news likelihood": "likelihood", "Move %": "move"}
    )
    data["status"] = data["Flagged"].map({"yes": "Flagged for a brief", "no": "Not flagged"})
    threshold = DEFAULT_POLICY.material_event_threshold
    move = DEFAULT_POLICY.price_move_threshold_pct
    y_max = max(move + 1.5, float(data["move"].abs().max()) + 0.5) if len(data) else move + 1.5
    x = alt.X("likelihood:Q", title="Major-news likelihood (Jev)", scale=alt.Scale(domain=[0, 1]),
              axis=alt.Axis(format="%", tickCount=5, grid=False))
    y = alt.Y("move:Q", title="Price move today (%)", scale=alt.Scale(domain=[-y_max, y_max], nice=False),
              axis=alt.Axis(tickCount=7))

    zones = pd.DataFrame([
        {"x0": threshold, "x1": 1.0, "y0": -y_max, "y1": y_max},
        {"x0": 0.0, "x1": threshold, "y0": move, "y1": y_max},
        {"x0": 0.0, "x1": threshold, "y0": -y_max, "y1": -move},
    ])
    shade = alt.Chart(zones).mark_rect(color=FLAGGED_COLOR, opacity=0.08).encode(
        x="x0:Q", x2="x1:Q", y="y0:Q", y2="y1:Q"
    )
    labels = pd.DataFrame([
        {"x": threshold + 0.01, "y": y_max, "text": f"Major news likely (≥ {threshold:.0%})"},
        {"x": 0.01, "y": y_max, "text": f"Moved ≥ {move:g}% either way"},
    ])
    label_marks = alt.Chart(labels).mark_text(
        baseline="top", align="left", dy=6, fontSize=12, color=NOT_FLAGGED_COLOR
    ).encode(x="x:Q", y="y:Q", text="text:N")

    pick = alt.selection_point(name="pick", fields=["Ticker"], on="click")
    points = alt.Chart(data).mark_circle(size=70, stroke="white", strokeWidth=0.5).encode(
        x=x, y=y,
        color=alt.Color("status:N", title=None,
                        scale=alt.Scale(domain=["Flagged for a brief", "Not flagged"],
                                        range=[FLAGGED_COLOR, NOT_FLAGGED_COLOR]),
                        legend=alt.Legend(orient="top", direction="horizontal")),
        opacity=alt.condition(alt.datum.Flagged == "yes", alt.value(0.95), alt.value(0.45)),
        order=alt.Order("Flagged:N", sort="ascending"),  # flagged dots drawn on top
        tooltip=[
            alt.Tooltip("Company:N"), alt.Tooltip("Ticker:N"),
            alt.Tooltip("likelihood:Q", title="Major-news likelihood", format=".0%"),
            alt.Tooltip("move:Q", title="Move today (%)", format="+.2f"),
            alt.Tooltip("Why:N"),
        ],
    ).add_params(pick)

    chart = alt.layer(shade, label_marks, points).properties(height=420)
    if any(any(x.startswith(_EARLIER_POLICY_REASONS) for x in r.decision.reasons) for r in report.results):
        chart = chart.properties(title=alt.TitleParams(
            "This scan ran under an earlier, looser policy; the shading shows the current one.",
            fontSize=12, fontWeight="normal", anchor="start",
        ))
    return chart


_NO_SELECTION = {"selection": {"rows": [], "columns": [], "cells": []}}


def _on_table_pick(key: str, tickers: list[str], report_id: str) -> None:
    """Row or cell clicked: remember which company's pop-up to open."""
    sel = st.session_state[key]["selection"]
    rows = sel.get("rows") or [cell[0] for cell in sel.get("cells", [])]
    if rows and rows[0] < len(tickers):
        st.session_state["scan_open"] = (report_id, tickers[rows[0]])


def _on_chart_pick(key: str, report_id: str) -> None:
    picks = st.session_state[key]["selection"].get("pick") or []
    ticker = next((p.get("Ticker") for p in picks if p.get("Ticker")), None)
    if ticker:
        st.session_state["scan_open"] = (report_id, ticker)


def _scan_table(table: pd.DataFrame, key: str, report_id: str, height: int | str = "auto") -> None:
    """A clickable results table. Cell selection is on as well as the row
    checkbox, so clicking anywhere in a row opens it - most visitors click
    the company name, not the checkbox."""
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        height=height,
        on_select=partial(_on_table_pick, key, table["Ticker"].tolist(), report_id),
        selection_mode=["single-row", "single-cell"],
        key=key,
        column_config={
            "Move %": st.column_config.NumberColumn(format="%+.2f"),
            "Major-news likelihood": st.column_config.ProgressColumn(
                min_value=0.0, max_value=1.0, format="percent",
                help=f"Jev's probability that the headlines report {MAJOR_NEWS}; flagged at "
                     f"{DEFAULT_POLICY.material_event_threshold:.0%} or above",
            ),
            "Why": st.column_config.TextColumn(width="medium", help="Why the ticker was flagged for a full brief"),
        },
    )


def _render_report(report: ScanReport) -> None:
    st.markdown(_report_summary(report))
    with st.expander("How it works and how it was evaluated"):
        st.markdown(METHODOLOGY_MD)

    full_table = _results_table(report)
    if not any(r.brief or r.brief_error for r in report.results):
        full_table = full_table.drop(columns="Brief")  # always empty on Jev-only scans
    # Clicks are handled by callbacks that record which company to open; the
    # pop-up opens on that one run and the request is used up, so closing it
    # needs no rerun (a rerun on close jumped the page back to the top). The
    # table ticks are cleared before the tables draw, so a row can be clicked
    # again. The report (and the filter, below) are in the keys, so a row index
    # never carries over to a different table.
    report_id = report.started_at
    chart_key = f"scan_chart_{report_id}"
    flagged_key = f"scan_flagged_{report_id}"
    to_open = st.session_state.pop("scan_open", None)
    if to_open:
        for key in [k for k in st.session_state if str(k).startswith(("scan_flagged_", "scan_table_"))]:
            st.session_state[key] = _NO_SELECTION

    st.subheader("Where every company landed")
    st.caption("Each dot is one company. Dots in the shaded zones were flagged for a full brief. "
               "Hover for details, click to open it.")
    st.altair_chart(
        _scan_chart(full_table, report), width="stretch", selection_mode="pick", key=chart_key,
        on_select=partial(_on_chart_pick, chart_key, report_id),
    )

    flagged = full_table[full_table["Flagged"] == "yes"]
    st.subheader(f"Flagged for a full brief ({len(flagged)})")
    st.caption("Strongest signal first. Click a row to see the headlines behind it and the full brief.")
    _scan_table(
        flagged[["Ticker", "Company", "Why", "Major-news likelihood", "Move %"]],
        key=flagged_key, report_id=report_id, height=min(len(flagged), 10) * 35 + 38,
    )

    with st.expander(f"All {len(full_table)} companies"):
        show = st.segmented_control(
            "Show", ["All", "Flagged", "Not flagged"], default="All", key="scan_filter"
        ) or "All"
        table = full_table
        if show != "All":
            table = table[table["Flagged"] == ("yes" if show == "Flagged" else "no")]
        _scan_table(table, key=f"scan_table_{report_id}_{show}", report_id=report_id)

    if to_open and to_open[0] == report_id:
        _open_scan_detail(report, to_open[1])

    briefed = [r for r in report.results if r.brief or r.brief_error]
    if briefed:
        st.subheader("Briefs for flagged tickers")
        for r in briefed:
            with st.expander(f"{_display_name(r.ticker, r.company_name)} - "
                             f"{'; '.join(_plain_reason(x) for x in r.decision.reasons)}"):
                if r.brief:
                    st.markdown(_brief_md(r.brief.brief_markdown))
                    if r.brief.usage:
                        st.caption(f"Estimated Gemini cost: ${r.brief.usage.cost_usd:.4f}")
                else:
                    st.error(f"Brief failed: {r.brief_error}")


def _open_scan_detail(report: ScanReport, ticker: str) -> None:
    """Show one row's detail in a pop-up, so the brief isn't buried below the table."""
    result = next((r for r in report.results if r.ticker == ticker), None)
    title = _display_name(ticker, result.company_name if result else None)
    st.dialog(title, width="large")(_render_scan_detail)(report, ticker)


def _render_scan_detail(report: ScanReport, ticker: str) -> None:
    """The evidence behind one row: verdict, headlines Jev read, and the brief on request."""
    result = next((r for r in report.results if r.ticker == ticker), None)
    if result is None:
        return
    when = datetime.fromisoformat(report.started_at).strftime("%-d %b %Y")
    headlines = [h.model_dump() for h in result.headlines]
    caption = f"From the {when} scan."
    if not headlines:
        caption += " Headlines weren't recorded in this scan."
    _render_verdict(
        ticker, result.company_name, result.screen.model_dump() if result.screen else None,
        result.decision, result.pct_change, headlines, caption=caption, when=f"on {when}",
    )
    if result.brief:
        _compact_brief(result.brief.brief_markdown)
        st.caption("Brief generated during the scan.")
    else:
        material = result.screen.material_event if result.screen else None
        _brief_flow(ticker, result.decision.escalate, "scan_brief",
                    close_call=_is_close_call(material, result.decision.escalate), in_dialog=True)


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


def _screen_single(query: str) -> dict:
    """Step 1: the nightly scan's check for one ticker - quote, headlines, Jev, policy.

    Jev reads the same unfiltered headlines as the nightly scan, so the verdict
    is exactly what tonight's scan would decide; only the display below picks
    out the on-topic headlines.
    """
    started = time.perf_counter()
    ticker = resolve_ticker(query)
    quote = get_quote_snapshot(ticker)
    news = get_company_news(ticker)
    result = {"query": query, "ticker": ticker, "quote": quote, "news": news, "screen": None, "decision": None,
              "error": None, "not_found": False, "cost_usd": 0.0}
    if not quote["success"] or not news["success"]:
        errors = [d["error"] for d in (quote, news) if not d["success"]]
        result["error"] = "; ".join(errors)
        # Both tools phrase an unknown symbol this way; anything else (network,
        # rate limits) is a real failure worth showing as-is.
        result["not_found"] = all("found for ticker" in e for e in errors)
    else:
        screen = screen_ticker(ticker, quote, news)
        result["screen"] = screen
        if screen.get("success"):
            result["cost_usd"] = screen["usage"]["cost_usd"]
            result["decision"] = decide(screen, quote.get("pct_change"), DEFAULT_POLICY)
        else:
            result["error"] = screen["error"]
    result["seconds"] = time.perf_counter() - started
    return result


MAJOR_NEWS = "earnings, guidance, M&A, lawsuits, regulation or a leadership change"
# Not flagged, but near enough the cut-off that "no sign of major news" would
# be untrue - e.g. 59% against a 60% threshold.
CLOSE_CALL_FLOOR = 0.45


def _is_close_call(material_event: float | None, flagged: bool) -> bool:
    return not flagged and material_event is not None and material_event >= CLOSE_CALL_FLOOR


def _display_name(ticker: str, fallback: str | None = None) -> str:
    """'Apple Inc. (AAPL)' when the name is known, else just the ticker."""
    name = _company_names().get(ticker) or fallback
    return f"{name} ({ticker})" if name else ticker


def _render_verdict(
    ticker: str,
    company_name: str | None,
    screen: dict | None,
    decision,
    move: float | None,
    headlines: list[dict],
    caption: str,
    when: str = "on the latest trading day",
) -> None:
    """The screening verdict for one company - used by Single ticker and the
    Daily Scan detail panel. screen is a ScreenResult as a dict, or None if
    screening failed."""
    name = _display_name(ticker, company_name)
    threshold = DEFAULT_POLICY.material_event_threshold
    move_limit = DEFAULT_POLICY.price_move_threshold_pct
    with st.container(border=True):
        if decision.escalate:
            st.markdown(f"### {name}: flagged for a full brief")
            why = []
            if screen and screen["material_event"] >= threshold:
                why.append(f"Recent headlines suggest major news ({MAJOR_NEWS}).")
            if move is not None and abs(move) >= move_limit:
                why.append(f"The stock moved {move:+.1f}% {when}.")
            if not screen:
                why.append("Screening couldn't run, so it was flagged to be safe.")
            st.markdown("\n".join(f"- {w}" for w in why) or "- Flagged by the screening rules at the time.")
        elif _is_close_call(screen["material_event"] if screen else None, False):
            st.markdown(f"### {name}: close call")
            st.markdown(
                f"Major-news likelihood of {screen['material_event']:.0%} is just below the "
                f"{threshold:.0%} cut-off, so the scan doesn't flag it - but the headlines below may "
                "still be worth a closer look."
            )
        else:
            st.markdown(f"### {name}: nothing major detected")
            st.markdown(f"The screen didn't detect major news ({MAJOR_NEWS}), and there was no big price move {when}.")

        cols = st.columns(3)
        if screen:
            cols[0].metric(
                "Major-news likelihood", f"{screen['material_event']:.0%}",
                help=f"Jev's probability that the headlines report {MAJOR_NEWS}. "
                     f"Tickers at {threshold:.0%} or above are flagged for a full brief.",
            )
            cols[1].metric(
                "News tone", screen["sentiment"],
                help=f"Jev's confidence in this call: {screen['sentiment_confidence']:.0%}",
            )
        cols[2].metric(
            "Price move", f"{round(move, 2) + 0.0:+.2f}%" if move is not None else "n/a",  # + 0.0: no "-0.00%"
            help=f"Moves of {move_limit:.0f}% or more are flagged for a full brief.",
        )

        about = headlines_about(headlines, ticker, company_name or _company_names().get(ticker, ""))
        if about:
            st.markdown("**Headlines about the company**")
            for h in about[:5]:
                st.markdown(f"- [{_md(h['headline'])}]({h['url']}) - {h['source']}, {h['published_at'][:10]}")
        elif headlines:
            st.markdown("**Headlines the screen read**")
            for h in headlines[:5]:
                st.markdown(f"- [{_md(h['headline'])}]({h['url']}) - {h['source']}, {h['published_at'][:10]}")
        st.caption(caption)


def _render_screen(result: dict) -> None:
    ticker = result["ticker"]
    if result.get("not_found"):
        st.warning(
            f"We couldn't find a US-listed company matching \"{result['query']}\". "
            "This covers US-listed companies only - international listings (e.g. Samsung) "
            "and private companies aren't supported. Try a ticker such as AAPL, or a company name such as Apple."
        )
        return
    if result["error"]:
        st.error(f"Couldn't screen {ticker} right now: {result['error']}")
        return
    _render_verdict(
        ticker, result["news"].get("company_name"), result["screen"], result["decision"],
        result["quote"].get("pct_change"), result["news"]["headlines"],
        caption=f"Screened by Jev in {result['seconds']:.1f} s.",
    )


def _stored_brief(state_key: str, ticker: str) -> dict | None:
    stored = st.session_state.get(state_key)
    return stored if isinstance(stored, dict) and stored.get("ticker") == ticker else None


def _brief_flow(ticker: str, flagged: bool, state_key: str, close_call: bool = False,
                in_dialog: bool = False) -> None:
    """The "Get the full research brief" button, its progress box and the brief.

    state_key keeps each place's brief separate (Single ticker vs. Daily Scan).
    in_dialog reruns only the pop-up once the brief is ready, so it stays open.
    """
    done = _stored_brief(state_key, ticker) is not None
    with st.container(horizontal=True, vertical_alignment="center"):
        want_brief = st.button(
            f"Get the full research brief ({BRIEF_TIME_HINT})",
            type="primary" if flagged or close_call else "secondary",
            disabled=done,
            key=f"{state_key}_button",
        )
        st.caption(
            "Recommended - the screen flagged this ticker." if flagged
            else "Worth a look - a close call." if close_call
            else "Nothing was flagged, but you can still run the full brief."
        )
    if want_brief:
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

            def on_retry(attempt):
                progress.update(research_done=False, sentiment_done=False, analyst_started=False)
                status.update(label="Taking longer than usual - retrying...")
                status.write(f"⏳ The model service was slow, so the brief restarted (attempt {attempt}).")

            started = time.perf_counter()
            try:
                brief = asyncio.run(run_research_brief(ticker, on_event=on_event, on_retry=on_retry))
            except BriefTimeoutError:
                status.update(label="Timed out", state="error")
                st.warning(
                    "The research service is responding slowly right now - usually because too many "
                    f"briefs were requested in the last minute - so the brief was stopped after "
                    f"{BRIEF_TIMEOUT_SECONDS * BRIEF_ATTEMPTS // 60} minutes. The screening result above "
                    "still stands - please try the brief again in a minute."
                )
            except Exception as exc:
                status.update(label="Failed", state="error")
                st.error(
                    "Something went wrong while generating the research brief. "
                    "This is usually a missing/invalid API key or a temporary rate limit. "
                    f"Details: {exc}"
                )
            else:
                status.update(label="Research brief ready", state="complete")
                st.session_state[state_key] = {
                    "ticker": ticker, "brief": brief, "seconds": time.perf_counter() - started,
                }
        if _stored_brief(state_key, ticker):
            # Redraw with the button disabled and the brief below.
            st.rerun(scope="fragment" if in_dialog else "app")

    stored = _stored_brief(state_key, ticker)
    if stored:
        brief, seconds = stored["brief"], stored["seconds"]
        if not brief.research.get("success", True):
            st.warning(f"Quantitative data unavailable for {brief.ticker}: {brief.research.get('error')}")
        if not brief.sentiment.get("success", True):
            st.warning(f"Sentiment data unavailable for {brief.ticker}: {brief.sentiment.get('error')}")

        if in_dialog:
            _compact_brief(brief.brief_markdown)
        else:
            with st.container(border=True):
                st.markdown(_brief_md(brief.brief_markdown))
        st.caption(f"Brief generated in {seconds:.0f} s.")

        with st.expander("Technical details"):
            st.caption("What each data step returned, before the Analyst wrote the brief.")
            st.markdown("**Research step** (price and fundamentals from Finnhub)")
            st.json(brief.research, expanded=False)
            st.markdown("**Sentiment Agent** (news tone and the headlines it cited)")
            st.json(brief.sentiment, expanded=False)


def _latest_index_report() -> tuple[ScanReport, bool] | None:
    """(report, is_sample) for the newest S&P 500 scan - the nightly job's
    universe - so an ad-hoc S&P 100 or live scan never becomes the headline;
    the newest S&P 100 scan only if no S&P 500 scan exists yet."""
    reports = [(_load_report(str(p), p.stat().st_mtime), is_demo_report(p)) for p in list_reports()]
    for universe in ("sp500", "sp100"):
        match = next((r for r in reports if r[0].universe == universe), None)
        if match:
            return match
    return None


# The nightly job runs Tue-Sat at 03:00 IST, so the longest normal gap is
# Saturday to Tuesday - 3 days. Past this, a run was probably missed.
STALE_SCAN_HOURS = 84


def _scan_age(finished_at: str) -> tuple[str, bool]:
    """('2 days ago', is_stale) for a scan's finish time."""
    hours = (datetime.now(timezone.utc) - datetime.fromisoformat(finished_at)).total_seconds() / 3600
    if hours < 1:
        age = "just now"
    elif hours < 24:
        age = f"{hours:.0f} hour{'s' if round(hours) != 1 else ''} ago"
    else:
        days = hours / 24
        age = f"{days:.0f} day{'s' if round(days) != 1 else ''} ago"
    return age, hours > STALE_SCAN_HOURS


def _render_last_scan_summary() -> None:
    """The result up front: what the last nightly scan did, and why it can be trusted."""
    latest = _latest_index_report()
    if not latest:
        return
    report, is_sample = latest
    avoided = report.tickers_scanned - report.tickers_escalated
    when = datetime.fromisoformat(report.started_at).strftime("%-d %b %Y")
    universe = {"sp500": "S&P 500", "sp100": "S&P 100"}[report.universe]
    with st.container(border=True):
        cols = st.columns(4)
        age, stale = _scan_age(report.finished_at)
        if is_sample:  # a committed sample report: its age says nothing about the nightly job
            age, stale = "sample scan", False
        cols[0].metric(
            "Last nightly scan", when, delta=age, delta_color="inverse" if stale else "off", delta_arrow="off",
            help=f"{universe}, Jev screening only. The scan runs Tuesday-Saturday at 03:00 IST, after each "
                 "US trading day.",
        )
        cols[1].metric("Tickers screened", report.tickers_scanned)
        cols[2].metric("Flagged for a full brief", report.tickers_escalated)
        cols[3].metric(
            "Research briefs avoided", f"{avoided} ({avoided / report.tickers_scanned:.0%})",
            help="Tickers the screen cleared, so no expensive Gemini brief was needed",
        )
        st.caption(
            f"Screening cost ${report.jev_cost_usd:.3f} for the whole {universe}. On a held-out, hand-labeled test "
            f"set the same policy caught {HELD_OUT['caught']} of {HELD_OUT['events']} material events. "
            "Details in the Daily Scan tab."
        )
        if stale:
            st.warning(
                f"The latest scan is {age} - the nightly job may have missed a run (the Mac asleep or "
                "shut down, or an API problem). Its log is at ~/Library/Logs/finance-research-engine-scan.log."
            )


_render_last_scan_summary()

tab_brief, tab_scan = st.tabs(["Single ticker", "Daily Scan"])

with tab_brief:
    missing_keys = [k for k in ("GOOGLE_API_KEY", "FINNHUB_API_KEY") if not os.environ.get(k)]
    if missing_keys:
        st.warning(
            f"Missing configuration: {', '.join(missing_keys)}. "
            "Set these in your local .env file, or in Streamlit secrets when deployed."
        )

    st.caption(
        "Step 1 is a cheap Jev screen - the same check the nightly scan runs. "
        "Step 2, the full multi-agent research brief, runs only if you ask for it."
    )
    with st.form("ticker_form"):
        ticker_input = st.text_input("Stock ticker or company name", placeholder="e.g. Apple or AAPL")
        submitted = st.form_submit_button("Screen")

    with st.container(horizontal=True, horizontal_alignment="left", vertical_alignment="center", gap="small"):
        st.caption("Or try an example:", width="content")
        clicked = [  # a list, so every button is drawn
            t for t, name in EXAMPLE_TICKERS.items() if st.button(f"{name} ({t})", key=f"example_{t}")
        ]
    example = clicked[0] if clicked else None

    if submitted or example:
        query = example or (ticker_input or "").strip()
        st.session_state.pop("screen_result", None)
        st.session_state.pop("brief", None)
        if not query:
            st.warning("Please enter a ticker symbol.")
        else:
            with st.spinner(f"Screening {query} with Jev..."):
                st.session_state["screen_result"] = _screen_single(query)

    result = st.session_state.get("screen_result")
    if result:
        _render_screen(result)
    if result and not result.get("not_found"):  # no brief button for a company we couldn't find
        ticker = result["ticker"]
        escalate = result["decision"] is not None and result["decision"].escalate
        material = result["screen"]["material_event"] if result["screen"] and result["screen"].get("success") else None
        _brief_flow(ticker, escalate, "brief", close_call=_is_close_call(material, escalate))


with tab_scan:
    missing_scan_keys = [k for k in ("TYPESAFE_API_KEY", "FINNHUB_API_KEY") if not os.environ.get(k)]
    if missing_scan_keys:
        st.warning(f"Live scans need {', '.join(missing_scan_keys)}.")
    _render_live_scan()

    reports = list_reports()
    if not reports:
        st.info("No scans yet. Run `python run_scan.py --universe sp100 --no-escalate` to create one.")
    else:
        # Open on the same scan the top strip summarises, not whichever report is newest.
        latest = _latest_index_report()
        default = next((i for i, p in enumerate(reports) if latest and
                        _load_report(str(p), p.stat().st_mtime).started_at == latest[0].started_at), 0)
        selected = st.selectbox("Scan report", reports, index=default, format_func=_report_label)
        _render_report(_load_report(str(selected), selected.stat().st_mtime))
