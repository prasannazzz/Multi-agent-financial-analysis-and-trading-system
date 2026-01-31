"""Data fetchers for the TradingAgents framework."""

from .news_scraper import NewsScraper
from .stock_data import StockDataFetcher
from .market_data import MarketDataFetcher, fetch_market_data

__all__ = [
    "NewsScraper",
    "StockDataFetcher",
    "MarketDataFetcher",
    "fetch_market_data",
]
