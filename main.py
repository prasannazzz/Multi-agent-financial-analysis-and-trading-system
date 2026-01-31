#!/usr/bin/env python3
"""
TradingAgents - Production-Ready Multi-Agent Financial Analysis System

A LangGraph-powered trading analysis framework with multiple specialized
analyst agents that collaborate to produce trading recommendations.

Usage:
    python main.py AAPL                    # Analyze single ticker
    python main.py AAPL MSFT GOOGL         # Analyze multiple tickers
    python main.py --visualize             # Show workflow graph
    python main.py AAPL --output json      # Output as JSON
    python main.py AAPL --verbose          # Verbose logging
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("TradingAgents")


def setup_environment() -> bool:
    """Load environment variables and validate configuration."""
    load_dotenv()
    
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        logger.error("GROQ_API_KEY not found in environment")
        logger.error("Set it in .env file: GROQ_API_KEY=gsk_...")
        return False
    
    if groq_key.startswith("gsk_") and len(groq_key) > 20:
        logger.info("Groq API key configured ✓")
        return True
    
    logger.warning("GROQ_API_KEY format may be invalid")
    return True


def visualize_graph(save_path: Optional[str] = None) -> None:
    """
    Visualize the LangGraph workflow and optionally save to file.
    
    Args:
        save_path: Optional path to save the graph image
    """
    try:
        from analysts import AnalystsTeam
        
        logger.info("Building workflow graph...")
        team = AnalystsTeam()
        
        # Get the graph structure
        graph = team.graph
        
        # Try to use graphviz for visualization
        try:
            from IPython.display import Image, display
            
            # Generate PNG using mermaid
            png_data = graph.get_graph().draw_mermaid_png()
            
            if save_path:
                with open(save_path, "wb") as f:
                    f.write(png_data)
                logger.info(f"Graph saved to: {save_path}")
            else:
                # Save to default location
                output_path = Path("workflow_graph.png")
                with open(output_path, "wb") as f:
                    f.write(png_data)
                logger.info(f"Graph saved to: {output_path.absolute()}")
                
        except ImportError:
            # Fallback: print ASCII representation
            logger.info("Graphviz not available, printing ASCII representation:")
            print_ascii_graph()
            
    except Exception as e:
        logger.error(f"Failed to visualize graph: {e}")
        print_ascii_graph()


def print_ascii_graph() -> None:
    """Print ASCII representation of the workflow."""
    graph_ascii = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                        TRADING AGENTS WORKFLOW GRAPH                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║                              ┌─────────────┐                                 ║
║                              │   START     │                                 ║
║                              └──────┬──────┘                                 ║
║                                     │                                        ║
║                                     ▼                                        ║
║                         ┌───────────────────────┐                            ║
║                         │    NEWS ANALYST       │                            ║
║                         │  (Market News Impact) │                            ║
║                         └───────────┬───────────┘                            ║
║                                     │                                        ║
║                                     ▼                                        ║
║                         ┌───────────────────────┐                            ║
║                         │ FUNDAMENTALS ANALYST  │                            ║
║                         │ (Financial Reports)   │                            ║
║                         └───────────┬───────────┘                            ║
║                                     │                                        ║
║                                     ▼                                        ║
║                         ┌───────────────────────┐                            ║
║                         │  SENTIMENT ANALYST    │                            ║
║                         │ (Market Psychology)   │                            ║
║                         └───────────┬───────────┘                            ║
║                                     │                                        ║
║                                     ▼                                        ║
║                         ┌───────────────────────┐                            ║
║                         │  TECHNICAL ANALYST    │                            ║
║                         │ (Price Action/Charts) │                            ║
║                         └───────────┬───────────┘                            ║
║                                     │                                        ║
║                                     ▼                                        ║
║                         ┌───────────────────────┐                            ║
║                         │     CONSOLIDATE       │                            ║
║                         │   (CIO Decision)      │                            ║
║                         └───────────┬───────────┘                            ║
║                                     │                                        ║
║                                     ▼                                        ║
║                              ┌─────────────┐                                 ║
║                              │     END     │                                 ║
║                              └─────────────┘                                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(graph_ascii)


def analyze_ticker(
    ticker: str,
    verbose: bool = False,
    output_format: str = "text",
) -> Dict[str, Any]:
    """
    Run full analysis on a single ticker.
    
    Args:
        ticker: Stock ticker symbol
        verbose: Enable verbose logging
        output_format: "text" or "json"
        
    Returns:
        Analysis result dictionary
    """
    from analysts import AnalystsTeam
    from data import MarketDataFetcher
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Analyzing {ticker}...")
    
    # Fetch market data
    logger.info("Fetching market data...")
    fetcher = MarketDataFetcher(verbose=verbose)
    
    try:
        market_data = fetcher.fetch_market_data(
            ticker=ticker,
            news_days=3,
            news_limit=10,
            price_days=60,
        )
    except Exception as e:
        logger.error(f"Failed to fetch market data: {e}")
        return {"error": str(e), "ticker": ticker}
    
    # Log data summary
    logger.info(f"  Price: ${market_data.get('current_price', 0):.2f}")
    logger.info(f"  News articles: {len(market_data.get('news_articles', []))}")
    logger.info(f"  Price history: {len(market_data.get('price_history', []))} days")
    
    # Run analysis
    logger.info("Running analyst team...")
    team = AnalystsTeam()
    
    try:
        result = team.analyze(ticker=ticker, market_data=market_data)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return {"error": str(e), "ticker": ticker}
    
    # Add metadata
    result["ticker"] = ticker
    result["timestamp"] = datetime.now().isoformat()
    result["market_data_summary"] = {
        "current_price": market_data.get("current_price"),
        "news_count": len(market_data.get("news_articles", [])),
        "price_history_days": len(market_data.get("price_history", [])),
    }
    
    return result


def format_result_text(result: Dict[str, Any]) -> str:
    """Format analysis result as readable text."""
    if "error" in result:
        return f"❌ Error analyzing {result.get('ticker', 'unknown')}: {result['error']}"
    
    lines = [
        "",
        "═" * 70,
        f"  TRADING ANALYSIS: {result.get('ticker', 'N/A')}",
        f"  Generated: {result.get('timestamp', 'N/A')}",
        "═" * 70,
        "",
        f"  📊 RECOMMENDATION: {result.get('final_signal', 'N/A')}",
        f"  📈 Confidence:     {result.get('confidence', 0):.1%}",
        f"  💰 Position Size:  {result.get('position_size', 'N/A')}",
        f"  ⏱️  Time Horizon:   {result.get('time_horizon', 'N/A')}",
        "",
        "─" * 70,
        "  REASONING:",
        "─" * 70,
        f"  {result.get('reasoning', 'N/A')}",
        "",
        "─" * 70,
        "  ANALYST AGREEMENT:",
        "─" * 70,
        f"  {result.get('analyst_agreement', 'N/A')}",
        "",
    ]
    
    # Risk factors
    risk_factors = result.get("risk_factors", [])
    if risk_factors:
        lines.extend([
            "─" * 70,
            "  ⚠️  RISK FACTORS:",
            "─" * 70,
        ])
        for risk in risk_factors:
            lines.append(f"    • {risk}")
        lines.append("")
    
    # Individual analyst signals
    individual = result.get("individual_reports", {})
    if individual:
        lines.extend([
            "─" * 70,
            "  INDIVIDUAL ANALYST SIGNALS:",
            "─" * 70,
        ])
        for name, report in individual.items():
            if report:
                sig = report.get("signal", "N/A")
                conf = report.get("confidence", 0)
                lines.append(f"    {name.upper():15} {sig:6} ({conf:.0%})")
        lines.append("")
    
    lines.append("═" * 70)
    
    return "\n".join(lines)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="TradingAgents - Multi-Agent Financial Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py AAPL                    Analyze Apple stock
  python main.py AAPL MSFT GOOGL         Analyze multiple stocks
  python main.py --visualize             Show workflow graph
  python main.py AAPL --output json      Output as JSON
  python main.py AAPL -v                 Verbose mode

Environment:
  GROQ_API_KEY    Required. Get from https://console.groq.com/keys
        """,
    )
    
    parser.add_argument(
        "tickers",
        nargs="*",
        help="Stock ticker symbols to analyze (e.g., AAPL MSFT)",
    )
    parser.add_argument(
        "--visualize", "-g",
        action="store_true",
        help="Visualize the workflow graph",
    )
    parser.add_argument(
        "--output", "-o",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--save-graph",
        type=str,
        metavar="PATH",
        help="Save workflow graph to specified path",
    )
    
    args = parser.parse_args()
    
    # Setup environment
    if not setup_environment():
        return 1
    
    # Handle visualization
    if args.visualize or args.save_graph:
        visualize_graph(args.save_graph)
        if not args.tickers:
            return 0
    
    # Require tickers for analysis
    if not args.tickers:
        parser.print_help()
        print("\n❌ Error: No tickers provided. Use --visualize to see the graph.")
        return 1
    
    # Analyze each ticker
    results = []
    for ticker in args.tickers:
        ticker = ticker.strip().upper()
        result = analyze_ticker(
            ticker=ticker,
            verbose=args.verbose,
            output_format=args.output,
        )
        results.append(result)
        
        # Output result
        if args.output == "json":
            print(json.dumps(result, indent=2, default=str))
        else:
            print(format_result_text(result))
    
    # Summary for multiple tickers
    if len(results) > 1 and args.output == "text":
        print("\n" + "═" * 70)
        print("  SUMMARY")
        print("═" * 70)
        for r in results:
            if "error" not in r:
                print(f"  {r.get('ticker', 'N/A'):8} {r.get('final_signal', 'N/A'):6} ({r.get('confidence', 0):.0%})")
        print("═" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
