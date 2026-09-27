"""Shared structured-data schemas for the finance research engine.

Single source of truth for the shape of data returned by tools/stock_data_tool.py
and tools/news_tool.py, referenced in the agents' instructions, and used to
validate output in tests.
"""

from typing import Literal, Optional

from pydantic import BaseModel


class StockData(BaseModel):
    success: bool = True
    ticker: str
    price: Optional[float] = None
    pe_ratio: Optional[float] = None
    revenue_growth: Optional[float] = None
    pct_change: Optional[float] = None
    week52_high: Optional[float] = None
    week52_low: Optional[float] = None
    market_cap: Optional[float] = None


class StockDataError(BaseModel):
    success: bool = False
    ticker: str
    error: str


class Headline(BaseModel):
    headline: str
    source: str
    url: str
    published_at: str


class NewsData(BaseModel):
    success: bool = True
    ticker: str
    headlines: list[Headline] = []
    note: Optional[str] = None


class NewsDataError(BaseModel):
    success: bool = False
    ticker: str
    error: str


class SentimentAssessment(BaseModel):
    success: bool = True
    ticker: str
    sentiment: Literal["bullish", "bearish", "neutral"]
    cited_headlines: list[Headline] = []


class SentimentAssessmentError(BaseModel):
    success: bool = False
    ticker: str
    error: str


class Usage(BaseModel):
    """Token counts and cost for one or more model calls.

    cost_usd is exact for Jev when the provider reports it per call and
    an estimate for Gemini (token counts x list price, see pricing.py).
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cost_is_estimate: bool = False
    # What the provider reported charging, when it reports it. Results from
    # the earlier Vercel setup recorded 0 here while free credits applied.
    # cost_usd is the list price, which is what cost comparisons use.
    billed_cost_usd: Optional[float] = None


class ResearchBrief(BaseModel):
    ticker: str
    research: dict
    sentiment: dict
    brief_markdown: str
    usage: Optional[Usage] = None


class ScreenResult(BaseModel):
    """Jev's screening answers for one ticker."""

    success: bool = True
    ticker: str
    sentiment: Literal["bullish", "bearish", "neutral"]
    sentiment_probabilities: dict[str, float] = {}
    sentiment_confidence: float
    material_event: float
    needs_analysis: float
    model: str
    latency_ms: float
    # Calls needed to get an answer; >1 means rate limits or transient
    # upstream errors were retried.
    attempts: int = 1
    usage: Usage


class ScreenError(BaseModel):
    success: bool = False
    ticker: str
    error: str


class Decision(BaseModel):
    escalate: bool
    reasons: list[str] = []


class TickerScanResult(BaseModel):
    ticker: str
    price: Optional[float] = None
    pct_change: Optional[float] = None
    headline_count: int = 0
    # The exact inputs Jev saw, so a scan can be re-screened with reworded
    # questions or turned into a labeled eval snapshot later.
    headlines: list[Headline] = []
    screen: Optional[ScreenResult] = None
    screen_error: Optional[str] = None
    decision: Decision
    brief: Optional[ResearchBrief] = None
    brief_error: Optional[str] = None


class ScanReport(BaseModel):
    started_at: str
    finished_at: str
    universe: str
    tickers_scanned: int
    tickers_escalated: int
    escalation_rate: float
    # False for --no-escalate runs: decisions are recorded, no briefs run.
    escalation_enabled: bool = True
    # Escalated tickers can exceed this; the rest are recorded without briefs.
    max_briefs: Optional[int] = None
    briefs_generated: int = 0
    jev_model: Optional[str] = None
    jev_cost_usd: float
    # True when the provider reported no per-call cost and it was estimated
    # from tokens x list price (pricing.jev_cost).
    jev_cost_is_estimate: bool = False
    gemini_cost_usd_estimate: float
    # Baseline A from the plan: a full Gemini brief for every scanned ticker,
    # extrapolated from this run's mean brief cost. None when no brief ran.
    baseline_all_briefs_cost_usd_estimate: Optional[float] = None
    jev_latency_p50_ms: Optional[float] = None
    jev_latency_p95_ms: Optional[float] = None
    results: list[TickerScanResult]
