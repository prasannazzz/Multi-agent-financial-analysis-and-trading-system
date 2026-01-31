# TradingAgents

A production-ready multi-agent financial analysis system powered by LangGraph and Groq LLM.

## Features

- **Multi-Agent Architecture**: Four specialized analyst agents collaborate to produce trading recommendations
- **LangGraph Workflow**: Orchestrated pipeline with state management
- **Real-Time Data**: Fetches live news and stock data
- **Graph Visualization**: Visual representation of the workflow
- **Production CLI**: Full command-line interface with multiple output formats

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MarketDataFetcher                       │
│  (News + Stock Data)                                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      AnalystsTeam                           │
│  (LangGraph Workflow)                                       │
├─────────────────────────────────────────────────────────────┤
│  ┌────────┐ ┌──────────────┐ ┌───────────┐ ┌─────────────┐ │
│  │ News   │ │ Fundamentals │ │ Sentiment │ │ Technical   │ │
│  │Analyst │ │   Analyst    │ │  Analyst  │ │   Analyst   │ │
│  └────────┘ └──────────────┘ └───────────┘ └─────────────┘ │
│                         │                                   │
│                         ▼                                   │
│              Consolidated Report (CIO)                      │
│              BUY/SELL/HOLD + Confidence                     │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd hekronpy

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirement.txt
```

## Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_api_key_here
```

Get your Groq API key from: https://console.groq.com/keys

## Usage

### Command Line

```bash
# Analyze a single stock
python main.py AAPL

# Analyze multiple stocks
python main.py AAPL MSFT GOOGL

# Visualize the workflow graph
python main.py --visualize

# Output as JSON
python main.py AAPL --output json

# Verbose mode
python main.py AAPL -v

# Save graph to specific path
python main.py --save-graph my_graph.png
```

### Python API

```python
from analysts import AnalystsTeam
from data import MarketDataFetcher

# Initialize
fetcher = MarketDataFetcher()
team = AnalystsTeam()

# Fetch market data
market_data = fetcher.fetch_market_data("AAPL")

# Run analysis
result = team.analyze(ticker="AAPL", market_data=market_data)

print(f"Signal: {result['final_signal']}")
print(f"Confidence: {result['confidence']:.1%}")
```

### Flask API

```bash
python -m data.api
```

Endpoints:
- `GET /api/news?keyword=AAPL&days=7` - Fetch news
- `GET /api/stock/AAPL` - Get stock data with indicators
- `GET /api/market-data/AAPL` - Full market data for analysis
- `GET /api/health` - Health check

## Project Structure

```
hekronpy/
├── main.py                 # Production CLI entry point
├── requirement.txt         # Dependencies
├── .env                    # Environment variables
├── README.md
│
├── analysts/               # LangGraph Analyst Agents
│   ├── __init__.py
│   ├── state.py            # Shared state definitions
│   ├── news_analyst.py     # News-driven analysis
│   ├── fundamentals_analyst.py
│   ├── sentiment_analyst.py
│   ├── technical_analyst.py
│   └── team.py             # LangGraph coordinator
│
├── data/                   # Data Fetchers
│   ├── __init__.py
│   ├── news_scraper.py     # MoneyControl news scraper
│   ├── stock_data.py       # yfinance + indicators
│   ├── market_data.py      # Combined fetcher
│   └── api.py              # Flask API
│
└── workflow_graph.png      # Generated workflow visualization
```

## Analyst Agents

| Agent | Role | Data Sources |
|-------|------|--------------|
| **News Analyst** | Analyzes market news impact | News articles, headlines |
| **Fundamentals Analyst** | Evaluates financial health | P/E, EPS, balance sheet |
| **Sentiment Analyst** | Assesses market psychology | Price action, news tone |
| **Technical Analyst** | Price patterns & indicators | RSI, MACD, SMA, volume |

## Output Example

```
══════════════════════════════════════════════════════════════════════
  TRADING ANALYSIS: AAPL
  Generated: 2026-01-31T20:37:01
══════════════════════════════════════════════════════════════════════

  📊 RECOMMENDATION: HOLD
  📈 Confidence:     65.0%
  💰 Position Size:  HALF
  ⏱️  Time Horizon:   MEDIUM

──────────────────────────────────────────────────────────────────────
  INDIVIDUAL ANALYST SIGNALS:
──────────────────────────────────────────────────────────────────────
    NEWS            HOLD   (0%)
    FUNDAMENTALS    HOLD   (70%)
    SENTIMENT       BUY    (70%)
    TECHNICAL       HOLD   (60%)
══════════════════════════════════════════════════════════════════════
```

## Dependencies

- **langgraph** - Workflow orchestration
- **langchain-groq** - Groq LLM integration
- **yfinance** - Stock data
- **beautifulsoup4** - News scraping
- **flask** - REST API
- **pandas/numpy** - Data processing

## License

MIT
