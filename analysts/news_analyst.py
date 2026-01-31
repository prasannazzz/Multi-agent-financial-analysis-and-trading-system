"""News-Driven Analyst Agent.

Analyzes stock news and macroeconomic updates to predict price movements.
Based on approaches from Lopez-Lira & Tang (2023) and FinGPT methodologies.
"""

from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq

from .state import AnalystState, AnalystReport


NEWS_ANALYST_PROMPT = """You are a senior financial news analyst specializing in market-moving events.

TASK: Analyze the following news articles for {ticker} and determine their impact on stock price.

NEWS ARTICLES:
{news_articles}

CURRENT PRICE: ${current_price}

Analyze each news item for:
1. Relevance to the company's core business
2. Short-term vs long-term implications
3. Market sentiment shift potential
4. Comparison with sector/market trends

Respond in JSON format:
{{
    "signal": "BUY" | "SELL" | "HOLD",
    "confidence": <float 0.0-1.0>,
    "reasoning": "<detailed analysis>",
    "key_factors": ["<factor1>", "<factor2>", ...]
}}
"""


class NewsAnalyst:
    """News-driven analyst agent that processes market news for trading signals."""

    def __init__(self, llm: Optional[ChatGroq] = None):
        self.llm = llm or ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
        self.prompt = ChatPromptTemplate.from_template(NEWS_ANALYST_PROMPT)
        self.parser = JsonOutputParser()

    def analyze(self, state: AnalystState) -> dict:
        """Analyze news and return updated state with news analysis."""
        market_data = state.get("market_data", {})
        news_articles = market_data.get("news_articles", [])

        if not news_articles:
            return {
                "news_analysis": {
                    "analyst_type": "news",
                    "signal": "HOLD",
                    "confidence": 0.0,
                    "reasoning": "No news articles available for analysis.",
                    "key_factors": [],
                }
            }

        news_text = "\n\n".join(
            [f"[{i+1}] {article}" for i, article in enumerate(news_articles)]
        )

        chain = self.prompt | self.llm | self.parser

        try:
            result = chain.invoke({
                "ticker": state["ticker"],
                "news_articles": news_text,
                "current_price": market_data.get("current_price", "N/A"),
            })

            return {
                "news_analysis": {
                    "analyst_type": "news",
                    "signal": result.get("signal", "HOLD"),
                    "confidence": float(result.get("confidence", 0.5)),
                    "reasoning": result.get("reasoning", ""),
                    "key_factors": result.get("key_factors", []),
                }
            }
        except Exception as e:
            return {
                "news_analysis": {
                    "analyst_type": "news",
                    "signal": "HOLD",
                    "confidence": 0.0,
                    "reasoning": f"Analysis failed: {str(e)}",
                    "key_factors": [],
                }
            }

    def __call__(self, state: AnalystState) -> dict:
        """Make the analyst callable for LangGraph nodes."""
        return self.analyze(state)
