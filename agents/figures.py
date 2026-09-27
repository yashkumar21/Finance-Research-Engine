"""Deterministic formatting of the brief's numbers - no model involved.

The Analyst Agent must never compute or reformat a number (it would turn
0.8338 into "0.83%" as easily as "83%"), so everything numeric in a brief is
formatted here: a key-figures table and display strings the Analyst copies
verbatim. KeyFiguresAgent runs this as a pipeline step between the data
agents and the Analyst; insert_key_figures then puts the exact table into
the finished brief.
"""

import json
import re
from datetime import datetime
from typing import AsyncGenerator, Optional

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

KEY_FIGURES_MARKER = "[[KEY_FIGURES]]"


def _money(value: float) -> str:
    return f"${value:,.2f}"


def _market_cap(value: float) -> str:
    for size, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if value >= size:
            return f"${value / size:.2f}{suffix}"
    return _money(value)


def _range_position(price: float, low: float, high: float) -> str:
    if high <= low:
        return ""
    share = (price - low) / (high - low)
    if share >= 0.9:
        return "near the 52-week high"
    if share <= 0.1:
        return "near the 52-week low"
    return f"{share:.0%} of the way from the 52-week low to the high"


def figure_strings(research: dict) -> dict[str, str]:
    """Display strings for each available figure, keyed by label."""
    figures = {}
    price = research.get("price")
    if price is not None:
        figures["Price"] = _money(price)
    if research.get("pct_change") is not None:
        figures["Today's move"] = f"{research['pct_change']:+.2f}%"
    low, high = research.get("week52_low"), research.get("week52_high")
    if low is not None and high is not None:
        position = _range_position(price, low, high) if price is not None else ""
        figures["52-week range"] = f"{_money(low)} - {_money(high)}" + (f" ({position})" if position else "")
    if research.get("market_cap") is not None:
        figures["Market cap"] = _market_cap(research["market_cap"])
    if research.get("pe_ratio") is not None:
        figures["P/E (trailing 12 months)"] = f"{research['pe_ratio']:.2f}"
    if research.get("revenue_growth") is not None:
        figures["Revenue growth (year over year)"] = f"{research['revenue_growth']:+.1%}"
    return figures


def format_key_figures(research: dict) -> str:
    """The brief's key-figures section body: a markdown table, or why there isn't one."""
    if not research.get("success"):
        return f"Quantitative data is currently unavailable: {research.get('error', 'unknown error')}."
    figures = figure_strings(research)
    if not figures:
        return "No quantitative figures were available for this ticker."
    rows = ["| Figure | Value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in figures.items()]
    return "\n".join(rows)


def format_figures_for_analyst(research: dict) -> str:
    """The same figures as plain lines, for the Analyst to quote in its prose."""
    if not research.get("success"):
        return f"UNAVAILABLE - {research.get('error', 'unknown error')}"
    figures = figure_strings(research)
    return "\n".join(f"- {k}: {v}" for k, v in figures.items()) or "NONE AVAILABLE"


def format_as_of(generated_at: str) -> str:
    """ISO timestamp -> '27 Sep 2026, 16:02 UTC'."""
    return datetime.fromisoformat(generated_at).strftime("%d %b %Y, %H:%M UTC").lstrip("0")


def insert_key_figures(brief_markdown: str, key_figures: str) -> str:
    """Make the brief's Key figures section exactly the formatted table.

    Whatever the Analyst wrote under "## Key figures" - the marker, or its
    own rendition of the numbers - is replaced, so the section never depends
    on the model copying numbers faithfully. If the heading is missing, the
    section goes before the news section (or after the title).
    """
    heading = "## Key figures"
    start = brief_markdown.find(heading)
    if start != -1:
        body_start = start + len(heading)
        next_heading = brief_markdown.find("\n## ", body_start)
        end = next_heading + 1 if next_heading != -1 else len(brief_markdown)
        return brief_markdown[:body_start] + f"\n\n{key_figures}\n\n" + brief_markdown[end:]
    if KEY_FIGURES_MARKER in brief_markdown:
        return brief_markdown.replace(KEY_FIGURES_MARKER, key_figures, 1)
    section = f"{heading}\n\n{key_figures}\n\n"
    news_heading = "## What's in the news"
    if news_heading in brief_markdown:
        return brief_markdown.replace(news_heading, section + news_heading, 1)
    title_end = brief_markdown.find("\n")
    if title_end == -1:
        return brief_markdown + "\n\n" + section
    return brief_markdown[: title_end + 1] + "\n" + section + brief_markdown[title_end + 1 :]


def correct_citation_dates(brief_markdown: str, sentiment: dict) -> str:
    """Set each cited link's date to the article's real publication date.

    The Analyst copies dates from published_at and has been seen to get the
    year wrong, so every "[..](url) - source, YYYY-MM-DD" is checked against
    the article it links to.
    """
    for h in sentiment.get("cited_headlines") or []:
        url, published = h.get("url"), (h.get("published_at") or "")[:10]
        if not url or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", published):
            continue
        pattern = re.compile(r"(\]\(" + re.escape(url) + r"\)[^\n]*?)\d{4}-\d{2}-\d{2}")
        brief_markdown = pattern.sub(lambda m: m.group(1) + published, brief_markdown, count=1)
    return brief_markdown


def _parse(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[4:] if text.startswith("json") else text
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {"success": False, "error": "the Research Agent returned unreadable output"}


class KeyFiguresAgent(BaseAgent):
    """Pipeline step: formats research_data into key_figures / figures_text state."""

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        research = _parse(ctx.session.state.get("research_data"))
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            actions=EventActions(
                state_delta={
                    "key_figures": format_key_figures(research),
                    "figures_text": format_figures_for_analyst(research),
                }
            ),
        )


def build_key_figures_agent(name: Optional[str] = None) -> KeyFiguresAgent:
    return KeyFiguresAgent(name=name or "key_figures")
