"""Stock data fetcher with technical indicators using yfinance."""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None


class StockDataFetcher:
    """Fetches stock data and calculates technical indicators for LangGraph agents."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        if yf is None:
            raise ImportError("yfinance is required. Install with: pip install yfinance")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[StockData] {msg}")

    def get_historical_data(
        self,
        symbol: str,
        days: int = 365,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data."""
        try:
            ticker = yf.Ticker(symbol)
            end_date = datetime.today()
            start_date = end_date - timedelta(days=days)
            hist = ticker.history(start=start_date, end=end_date, interval="1d")
            self._log(f"Fetched {len(hist)} days of data for {symbol}")
            return hist
        except Exception as e:
            self._log(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()

    def get_company_info(self, symbol: str) -> dict:
        """Get company fundamentals."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return {
                "name": info.get("shortName", symbol),
                "symbol": symbol,
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "dividend_yield": info.get("dividendYield"),
                "eps": info.get("trailingEps"),
                "beta": info.get("beta"),
                "52_week_high": info.get("fiftyTwoWeekHigh"),
                "52_week_low": info.get("fiftyTwoWeekLow"),
                "avg_volume": info.get("averageVolume"),
                "description": info.get("longBusinessSummary", ""),
            }
        except Exception as e:
            self._log(f"Error getting info for {symbol}: {e}")
            return {"symbol": symbol, "error": str(e)}

    def calculate_sma(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return prices.rolling(window=window).mean()

    def calculate_ema(self, prices: pd.Series, span: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return prices.ewm(span=span, adjust=False).mean()

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Dict[str, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return {
            "macd_line": macd_line,
            "signal_line": signal_line,
            "histogram": histogram,
        }

    def calculate_bollinger_bands(
        self,
        prices: pd.Series,
        window: int = 20,
        num_std: int = 2,
    ) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)

        return {
            "upper_band": upper,
            "middle_band": sma,
            "lower_band": lower,
        }

    def get_technical_indicators(self, symbol: str, days: int = 365) -> dict:
        """Get all technical indicators for a symbol."""
        hist = self.get_historical_data(symbol, days)
        if hist.empty:
            return {"error": "No data available"}

        closes = hist["Close"].dropna()
        latest_price = float(closes.iloc[-1]) if len(closes) > 0 else 0

        # Calculate indicators
        rsi = self.calculate_rsi(closes)
        macd = self.calculate_macd(closes)
        bb = self.calculate_bollinger_bands(closes)

        # Get latest values
        latest_rsi = float(rsi.iloc[-1]) if len(rsi) > 0 and not np.isnan(rsi.iloc[-1]) else None
        latest_macd = float(macd["macd_line"].iloc[-1]) if len(macd["macd_line"]) > 0 else None
        latest_signal = float(macd["signal_line"].iloc[-1]) if len(macd["signal_line"]) > 0 else None

        # Moving averages
        sma_10 = float(self.calculate_sma(closes, 10).iloc[-1]) if len(closes) >= 10 else None
        sma_20 = float(self.calculate_sma(closes, 20).iloc[-1]) if len(closes) >= 20 else None
        sma_50 = float(self.calculate_sma(closes, 50).iloc[-1]) if len(closes) >= 50 else None
        sma_200 = float(self.calculate_sma(closes, 200).iloc[-1]) if len(closes) >= 200 else None

        # Momentum
        momentum_5d = ((closes.iloc[-1] - closes.iloc[-5]) / closes.iloc[-5] * 100) if len(closes) >= 5 else None
        momentum_20d = ((closes.iloc[-1] - closes.iloc[-20]) / closes.iloc[-20] * 100) if len(closes) >= 20 else None

        return {
            "current_price": latest_price,
            "rsi_14": latest_rsi,
            "macd": latest_macd,
            "macd_signal": latest_signal,
            "sma_10": sma_10,
            "sma_20": sma_20,
            "sma_50": sma_50,
            "sma_200": sma_200,
            "momentum_5d": momentum_5d,
            "momentum_20d": momentum_20d,
            "bb_upper": float(bb["upper_band"].iloc[-1]) if not np.isnan(bb["upper_band"].iloc[-1]) else None,
            "bb_lower": float(bb["lower_band"].iloc[-1]) if not np.isnan(bb["lower_band"].iloc[-1]) else None,
        }

    def get_price_history(self, symbol: str, days: int = 60) -> List[float]:
        """Get list of closing prices for LangGraph technical analyst."""
        hist = self.get_historical_data(symbol, days)
        if hist.empty:
            return []
        return hist["Close"].dropna().tolist()

    def get_volume_history(self, symbol: str, days: int = 60) -> List[int]:
        """Get list of volumes for LangGraph technical analyst."""
        hist = self.get_historical_data(symbol, days)
        if hist.empty:
            return []
        return hist["Volume"].dropna().astype(int).tolist()

    def get_financial_reports(self, symbol: str) -> dict:
        """Get financial report data for LangGraph fundamentals analyst."""
        info = self.get_company_info(symbol)
        indicators = self.get_technical_indicators(symbol)

        return {
            "income_statement": {
                "market_cap": f"${info.get('market_cap', 0) / 1e9:.2f}B" if info.get('market_cap') else "N/A",
                "pe_ratio": info.get("pe_ratio"),
                "forward_pe": info.get("forward_pe"),
                "eps": info.get("eps"),
            },
            "valuation": {
                "52_week_high": info.get("52_week_high"),
                "52_week_low": info.get("52_week_low"),
                "current_price": indicators.get("current_price"),
                "beta": info.get("beta"),
            },
            "dividends": {
                "dividend_yield": f"{info.get('dividend_yield', 0) * 100:.2f}%" if info.get('dividend_yield') else "N/A",
            },
            "company": {
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "description": info.get("description", "")[:500],
            },
        }
