"""Shared state definitions for the Analysts Team."""

from typing import TypedDict, List, Optional, Literal
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MarketData:
    """Container for market data inputs."""
    ticker: str
    current_price: float
    price_history: List[float] = field(default_factory=list)
    volume_history: List[int] = field(default_factory=list)
    news_articles: List[str] = field(default_factory=list)
    financial_reports: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class AnalystReport(TypedDict):
    """Individual analyst report structure."""
    analyst_type: str
    signal: Literal["BUY", "SELL", "HOLD"]
    confidence: float  # 0.0 to 1.0
    reasoning: str
    key_factors: List[str]


class AnalystState(TypedDict):
    """Shared state for the analysts team graph."""
    ticker: str
    market_data: dict
    news_analysis: Optional[AnalystReport]
    fundamentals_analysis: Optional[AnalystReport]
    sentiment_analysis: Optional[AnalystReport]
    technical_analysis: Optional[AnalystReport]
    consolidated_report: Optional[dict]
    messages: List[dict]
