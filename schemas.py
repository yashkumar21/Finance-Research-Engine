"""Shared structured-data schemas for the finance research engine.

Single source of truth for the shape of data returned by tools/stock_data_tool.py,
referenced in the Research Agent's instruction, and used to validate output in tests.
"""

from typing import Optional

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
