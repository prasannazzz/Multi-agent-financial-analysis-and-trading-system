"""Analysts Team Coordinator.

Orchestrates multiple analyst agents using LangGraph to produce
consolidated trading recommendations through parallel analysis and debate.
"""

from typing import Optional, Literal
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq

from .state import AnalystState
from .news_analyst import NewsAnalyst
from .fundamentals_analyst import FundamentalsAnalyst
from .sentiment_analyst import SentimentAnalyst
from .technical_analyst import TechnicalAnalyst


CONSOLIDATION_PROMPT = """You are the Chief Investment Officer consolidating reports from your analyst team.

TICKER: {ticker}

ANALYST REPORTS:

1. NEWS ANALYST:
   Signal: {news_signal} (Confidence: {news_confidence})
   Reasoning: {news_reasoning}
   Key Factors: {news_factors}

2. FUNDAMENTALS ANALYST:
   Signal: {fundamentals_signal} (Confidence: {fundamentals_confidence})
   Reasoning: {fundamentals_reasoning}
   Key Factors: {fundamentals_factors}

3. SENTIMENT ANALYST:
   Signal: {sentiment_signal} (Confidence: {sentiment_confidence})
   Reasoning: {sentiment_reasoning}
   Key Factors: {sentiment_factors}

4. TECHNICAL ANALYST:
   Signal: {technical_signal} (Confidence: {technical_confidence})
   Reasoning: {technical_reasoning}
   Key Factors: {technical_factors}

Your task:
1. Weigh each analyst's input based on confidence and reasoning quality
2. Identify agreements and conflicts between analysts
3. Synthesize a final recommendation with position sizing guidance

Respond in JSON format:
{{
    "final_signal": "BUY" | "SELL" | "HOLD",
    "confidence": <float 0.0-1.0>,
    "position_size": "FULL" | "HALF" | "QUARTER" | "NONE",
    "reasoning": "<consolidated analysis>",
    "risk_factors": ["<risk1>", "<risk2>", ...],
    "analyst_agreement": "<description of consensus/disagreement>",
    "time_horizon": "SHORT" | "MEDIUM" | "LONG"
}}
"""


class AnalystsTeam:
    """Coordinates multiple analyst agents using LangGraph."""

    def __init__(self, llm: Optional[ChatGroq] = None):
        self.llm = llm or ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)

        # Initialize analyst agents
        self.news_analyst = NewsAnalyst(llm=self.llm)
        self.fundamentals_analyst = FundamentalsAnalyst(llm=self.llm)
        self.sentiment_analyst = SentimentAnalyst(llm=self.llm)
        self.technical_analyst = TechnicalAnalyst(llm=self.llm)

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow for the analysts team."""

        # Define the graph with AnalystState
        workflow = StateGraph(AnalystState)

        # Add analyst nodes
        workflow.add_node("news_analyst", self.news_analyst)
        workflow.add_node("fundamentals_analyst", self.fundamentals_analyst)
        workflow.add_node("sentiment_analyst", self.sentiment_analyst)
        workflow.add_node("technical_analyst", self.technical_analyst)
        workflow.add_node("consolidate", self._consolidate_reports)

        # Set entry point - all analysts run in parallel from start
        workflow.set_entry_point("news_analyst")

        # Parallel execution: all analysts feed into consolidation
        workflow.add_edge("news_analyst", "fundamentals_analyst")
        workflow.add_edge("fundamentals_analyst", "sentiment_analyst")
        workflow.add_edge("sentiment_analyst", "technical_analyst")
        workflow.add_edge("technical_analyst", "consolidate")
        workflow.add_edge("consolidate", END)

        return workflow.compile()

    def _consolidate_reports(self, state: AnalystState) -> dict:
        """Consolidate all analyst reports into final recommendation."""

        news = state.get("news_analysis") or {}
        fundamentals = state.get("fundamentals_analysis") or {}
        sentiment = state.get("sentiment_analysis") or {}
        technical = state.get("technical_analysis") or {}

        prompt = ChatPromptTemplate.from_template(CONSOLIDATION_PROMPT)
        parser = JsonOutputParser()
        chain = prompt | self.llm | parser

        try:
            result = chain.invoke({
                "ticker": state["ticker"],
                "news_signal": news.get("signal", "N/A"),
                "news_confidence": news.get("confidence", 0),
                "news_reasoning": news.get("reasoning", "N/A"),
                "news_factors": ", ".join(news.get("key_factors", [])),
                "fundamentals_signal": fundamentals.get("signal", "N/A"),
                "fundamentals_confidence": fundamentals.get("confidence", 0),
                "fundamentals_reasoning": fundamentals.get("reasoning", "N/A"),
                "fundamentals_factors": ", ".join(fundamentals.get("key_factors", [])),
                "sentiment_signal": sentiment.get("signal", "N/A"),
                "sentiment_confidence": sentiment.get("confidence", 0),
                "sentiment_reasoning": sentiment.get("reasoning", "N/A"),
                "sentiment_factors": ", ".join(sentiment.get("key_factors", [])),
                "technical_signal": technical.get("signal", "N/A"),
                "technical_confidence": technical.get("confidence", 0),
                "technical_reasoning": technical.get("reasoning", "N/A"),
                "technical_factors": ", ".join(technical.get("key_factors", [])),
            })

            return {
                "consolidated_report": {
                    "final_signal": result.get("final_signal", "HOLD"),
                    "confidence": float(result.get("confidence", 0.5)),
                    "position_size": result.get("position_size", "NONE"),
                    "reasoning": result.get("reasoning", ""),
                    "risk_factors": result.get("risk_factors", []),
                    "analyst_agreement": result.get("analyst_agreement", ""),
                    "time_horizon": result.get("time_horizon", "MEDIUM"),
                    "individual_reports": {
                        "news": news,
                        "fundamentals": fundamentals,
                        "sentiment": sentiment,
                        "technical": technical,
                    },
                }
            }
        except Exception as e:
            return {
                "consolidated_report": {
                    "final_signal": "HOLD",
                    "confidence": 0.0,
                    "position_size": "NONE",
                    "reasoning": f"Consolidation failed: {str(e)}",
                    "risk_factors": ["Analysis error"],
                    "analyst_agreement": "Unable to consolidate",
                    "time_horizon": "MEDIUM",
                }
            }

    def analyze(self, ticker: str, market_data: dict) -> dict:
        """
        Run full analyst team analysis on a ticker.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
            market_data: Dictionary containing:
                - current_price: float
                - price_history: List[float]
                - volume_history: List[int]
                - news_articles: List[str]
                - financial_reports: dict

        Returns:
            Consolidated analysis report with trading recommendation.
        """
        initial_state: AnalystState = {
            "ticker": ticker,
            "market_data": market_data,
            "news_analysis": None,
            "fundamentals_analysis": None,
            "sentiment_analysis": None,
            "technical_analysis": None,
            "consolidated_report": None,
            "messages": [],
        }

        result = self.graph.invoke(initial_state)
        return result.get("consolidated_report", {})

    def get_individual_analysis(self, ticker: str, market_data: dict) -> dict:
        """Run analysis and return all individual analyst reports."""
        initial_state: AnalystState = {
            "ticker": ticker,
            "market_data": market_data,
            "news_analysis": None,
            "fundamentals_analysis": None,
            "sentiment_analysis": None,
            "technical_analysis": None,
            "consolidated_report": None,
            "messages": [],
        }

        result = self.graph.invoke(initial_state)

        return {
            "ticker": ticker,
            "news": result.get("news_analysis"),
            "fundamentals": result.get("fundamentals_analysis"),
            "sentiment": result.get("sentiment_analysis"),
            "technical": result.get("technical_analysis"),
            "consolidated": result.get("consolidated_report"),
        }
