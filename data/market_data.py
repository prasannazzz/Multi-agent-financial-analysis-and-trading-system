"""Unified market data fetcher for LangGraph TradingAgents."""

from typing import Dict, List, Any

from .news_scraper import NewsScraper
from .stock_data import StockDataFetcher


class MarketDataFetcher:
    """
    Unified data fetcher that combines news and stock data
    for the LangGraph AnalystsTeam.
    """

    def __init__(self, verbose: bool = False):
        self.news_scraper = NewsScraper(verbose=verbose)
        self.stock_fetcher = StockDataFetcher(verbose=verbose)
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[MarketData] {msg}")

    def fetch_market_data(
        self,
        ticker: str,
        news_keyword: str = "",
        news_days: int = 7,
        news_limit: int = 10,
        price_days: int = 60,
    ) -> Dict[str, Any]:
        """
        Fetch all market data needed for AnalystsTeam.analyze().

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL", "TCS.NS")
            news_keyword: Keyword to filter news (default: use ticker)
            news_days: Days of news to fetch
            news_limit: Max news articles
            price_days: Days of price history

        Returns:
            Dictionary matching AnalystsTeam market_data schema:
            {
                "current_price": float,
                "price_history": List[float],
                "volume_history": List[int],
                "news_articles": List[str],
                "financial_reports": dict,
            }
        """
        self._log(f"Fetching market data for {ticker}")

        # Use ticker as news keyword if not specified
        if not news_keyword:
            # Extract company name from ticker for better news matching
            news_keyword = ticker.split(".")[0]

        # Fetch news
        self._log("Fetching news articles...")
        news_articles = self.news_scraper.get_news_articles(
            keyword=news_keyword,
            num_days=news_days,
            limit=news_limit,
        )

        # Fetch stock data
        self._log("Fetching stock data...")
        indicators = self.stock_fetcher.get_technical_indicators(ticker)
        price_history = self.stock_fetcher.get_price_history(ticker, days=price_days)
        volume_history = self.stock_fetcher.get_volume_history(ticker, days=price_days)
        financial_reports = self.stock_fetcher.get_financial_reports(ticker)

        market_data = {
            "current_price": indicators.get("current_price", 0),
            "price_history": price_history,
            "volume_history": volume_history,
            "news_articles": news_articles,
            "financial_reports": financial_reports,
            "technical_indicators": indicators,
        }

        self._log(f"Fetched: {len(news_articles)} news, {len(price_history)} prices")
        return market_data

    def fetch_for_analysts(
        self,
        ticker: str,
        **kwargs,
    ) -> tuple:
        """
        Convenience method returning (ticker, market_data) tuple
        ready to pass to AnalystsTeam.analyze().

        Usage:
            fetcher = MarketDataFetcher()
            ticker, data = fetcher.fetch_for_analysts("AAPL")
            result = team.analyze(ticker=ticker, market_data=data)
        """
        market_data = self.fetch_market_data(ticker, **kwargs)
        return ticker, market_data

    def get_quick_summary(self, ticker: str) -> dict:
        """Get a quick summary for a ticker without full data fetch."""
        indicators = self.stock_fetcher.get_technical_indicators(ticker, days=30)
        info = self.stock_fetcher.get_company_info(ticker)

        return {
            "ticker": ticker,
            "name": info.get("name", ticker),
            "sector": info.get("sector"),
            "current_price": indicators.get("current_price"),
            "rsi": indicators.get("rsi_14"),
            "momentum_5d": indicators.get("momentum_5d"),
            "sma_20": indicators.get("sma_20"),
        }


def fetch_market_data(ticker: str, **kwargs) -> Dict[str, Any]:
    """Convenience function to fetch market data without instantiating class."""
    fetcher = MarketDataFetcher()
    return fetcher.fetch_market_data(ticker, **kwargs)
