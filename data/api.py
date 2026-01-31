"""Flask API endpoints for the TradingAgents data module."""

from flask import Flask, request, jsonify
from flask_cors import CORS

from .news_scraper import NewsScraper
from .stock_data import StockDataFetcher
from .market_data import MarketDataFetcher

app = Flask(__name__)
CORS(app)

# Fallback news data
FALLBACK_NEWS = [
    {
        "title": "Markets show mixed signals amid global uncertainty",
        "url": "#",
        "date": None,
        "source": "Fallback",
    }
]


@app.route("/api/news", methods=["GET"])
def get_news():
    """
    GET /api/news?keyword=TCS&days=7&limit=10

    Returns news articles as JSON.
    """
    keyword = request.args.get("keyword", "")
    days = int(request.args.get("days", 7))
    limit = int(request.args.get("limit", 15))

    try:
        scraper = NewsScraper()
        result = scraper.to_dict(keyword=keyword, num_days=days, limit=limit)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "articles": FALLBACK_NEWS}), 200


@app.route("/api/stock/<symbol>", methods=["GET"])
def get_stock_data(symbol: str):
    """
    GET /api/stock/AAPL

    Returns stock data with technical indicators.
    """
    try:
        fetcher = StockDataFetcher()
        indicators = fetcher.get_technical_indicators(symbol)
        info = fetcher.get_company_info(symbol)

        return jsonify({
            "symbol": symbol,
            "info": info,
            "indicators": indicators,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/market-data/<ticker>", methods=["GET"])
def get_market_data(ticker: str):
    """
    GET /api/market-data/AAPL?news_days=7&price_days=60

    Returns full market data for LangGraph analysts.
    """
    news_days = int(request.args.get("news_days", 7))
    price_days = int(request.args.get("price_days", 60))
    news_limit = int(request.args.get("news_limit", 10))

    try:
        fetcher = MarketDataFetcher()
        data = fetcher.fetch_market_data(
            ticker=ticker,
            news_days=news_days,
            price_days=price_days,
            news_limit=news_limit,
        )
        return jsonify({"ticker": ticker, "market_data": data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


def run_api(port: int = 5001, debug: bool = True):
    """Run the Flask API server."""
    app.run(port=port, debug=debug)


if __name__ == "__main__":
    run_api()
