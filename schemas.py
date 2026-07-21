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


class ResearchBrief(BaseModel):
    ticker: str
    research: dict
    sentiment: dict
    brief_markdown: str
