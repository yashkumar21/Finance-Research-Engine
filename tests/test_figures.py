"""Tests for agents.figures - the brief's numbers are formatted in Python, not by a model.

Pure unit tests: no network, no model calls.
"""

from agents.figures import (
    correct_citation_dates,
    figure_strings,
    format_as_of,
    format_key_figures,
    insert_key_figures,
)

NVDA = {
    "success": True, "ticker": "NVDA", "price": 225.07, "pe_ratio": 28.1222, "revenue_growth": 0.8338,
    "pct_change": 0.22, "week52_high": 236.54, "week52_low": 164.27, "market_cap": 5.42e12,
}


def test_figures_are_formatted_for_readers():
    f = figure_strings(NVDA)
    assert f["Price"] == "$225.07"
    assert f["Today's move"] == "+0.22%"
    assert f["Revenue growth (year over year)"] == "+83.4%"  # 0.8338 is a fraction
    assert f["Market cap"] == "$5.42T"
    assert f["P/E (trailing 12 months)"] == "28.12"
    assert f["52-week range"] == "$164.27 - $236.54 (84% of the way from the 52-week low to the high)"


def test_range_position_near_the_ends():
    assert "near the 52-week high" in figure_strings({**NVDA, "price": 235.0})["52-week range"]
    assert "near the 52-week low" in figure_strings({**NVDA, "price": 165.0})["52-week range"]


def test_missing_figures_are_left_out_not_invented():
    f = figure_strings({"success": True, "ticker": "X", "price": 10.0, "pe_ratio": None})
    assert set(f) == {"Price"}


def test_unavailable_data_says_so():
    text = format_key_figures({"success": False, "ticker": "X", "error": "No price data found"})
    assert "unavailable" in text and "No price data found" in text


def test_as_of_is_readable():
    assert format_as_of("2026-09-07T16:02:31.613591+00:00") == "7 Sep 2026, 16:02 UTC"


def test_key_figures_section_is_replaced_by_the_exact_table():
    brief = "# X\n## Bottom line\nb\n\n## Key figures\n- Price: $999\n\n## What's in the news\nn\n"
    out = insert_key_figures(brief, "TABLE")
    assert out.count("## Key figures") == 1
    assert "TABLE" in out and "$999" not in out
    assert out.index("TABLE") < out.index("## What's in the news")


def test_key_figures_section_is_added_when_missing():
    out = insert_key_figures("# X\n## Bottom line\nb\n## What's in the news\nn", "TABLE")
    assert out.index("## Key figures") < out.index("## What's in the news")


def test_citation_dates_are_corrected_to_the_article_date():
    sentiment = {"cited_headlines": [{"url": "https://x/a?id=1", "published_at": "2026-09-27T10:00:00+00:00"}]}
    brief = "- takeaway ([H](https://x/a?id=1) - Yahoo, 2027-09-27)"
    assert correct_citation_dates(brief, sentiment) == "- takeaway ([H](https://x/a?id=1) - Yahoo, 2026-09-27)"
