"""News scraper for financial news from Money Control."""

import random
import time
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup


class NewsScraper:
    """Scrapes financial news from Money Control for use in LangGraph agents."""

    def __init__(self, verbose: bool = False):
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }
        self.base_url = "https://www.moneycontrol.com/news/business/stocks"
        self.source = "Money Control"
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[NewsScraper] {msg}")

    def _get_soup(self, url: str) -> Optional[BeautifulSoup]:
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            self._log(f"Fetched: {url}")
            return BeautifulSoup(response.text, "html.parser")
        except requests.RequestException as e:
            self._log(f"Error fetching {url}: {e}")
            return None

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        if not date_str:
            return None

        date_str = date_str.lower().strip()
        now = datetime.now()

        try:
            if "min" in date_str:
                mins = int("".join(filter(str.isdigit, date_str)) or "0")
                return now - timedelta(minutes=mins)
            elif "hour" in date_str:
                hours = int("".join(filter(str.isdigit, date_str)) or "0")
                return now - timedelta(hours=hours)
            elif "day" in date_str:
                days = int("".join(filter(str.isdigit, date_str)) or "0")
                return now - timedelta(days=days)
            else:
                for fmt in ["%B %d, %Y %I:%M %p", "%b %d, %Y %I:%M %p", "%d %B %Y"]:
                    try:
                        return datetime.strptime(date_str, fmt)
                    except ValueError:
                        continue
        except Exception:
            pass
        return None

    def _extract_article(self, article) -> Optional[dict]:
        try:
            title_tag = (
                article.find("h2")
                or article.find("h1")
                or article.find("a", class_="headline")
            )
            if not title_tag:
                return None

            title = title_tag.get_text().strip()
            link_tag = title_tag.find("a") if title_tag.name != "a" else title_tag
            url = link_tag.get("href") if link_tag else None

            timestamp = (
                article.find("span", class_="article_schedule")
                or article.find("span", class_="date")
                or article.find("time")
            )
            date_str = timestamp.text.strip() if timestamp else None
            date = self._parse_date(date_str)

            self._log(f"Found: {title[:50]}...")

            return {
                "title": title,
                "url": url,
                "date": date,
                "source": self.source,
            }
        except Exception as e:
            self._log(f"Extract error: {e}")
            return None

    def scrape(
        self,
        keyword: str = "",
        num_days: int = 7,
        num_pages: int = 3,
        limit: int = 20,
    ) -> pd.DataFrame:
        """
        Scrape news articles.

        Args:
            keyword: Filter articles containing this keyword (empty = all)
            num_days: Only include articles from last N days
            num_pages: Number of pages to scrape
            limit: Max articles to return

        Returns:
            DataFrame with columns: title, url, date, source
        """
        all_articles = []
        cutoff = datetime.now() - timedelta(days=num_days)

        for page in range(1, num_pages + 1):
            url = f"{self.base_url}/page-{page}"
            soup = self._get_soup(url)
            if not soup:
                continue

            containers = (
                soup.find_all("li", class_="clearfix")
                or soup.find_all("div", class_="article-list")
                or soup.find_all("div", class_="article_box")
                or soup.find_all("div", class_="news_listing")
            )
            self._log(f"Page {page}: {len(containers)} items")

            for item in containers:
                data = self._extract_article(item)
                if not data:
                    continue

                # Keyword filter
                if keyword and keyword.lower() not in data["title"].lower():
                    continue

                # Date filter
                if data["date"] and data["date"] < cutoff:
                    continue

                all_articles.append(data)

            time.sleep(random.uniform(1, 2))

        df = pd.DataFrame(all_articles)
        if not df.empty:
            df = df.sort_values("date", ascending=False, na_position="last")
            df = df.head(limit)
        return df

    def get_news_articles(
        self,
        keyword: str = "",
        num_days: int = 7,
        num_pages: int = 2,
        limit: int = 15,
    ) -> List[str]:
        """
        Get news as list of formatted strings for LangGraph news node.

        Returns:
            List of news article strings ready for LLM consumption.
        """
        df = self.scrape(keyword, num_days, num_pages, limit)

        articles = []
        for _, row in df.iterrows():
            parts = [row.get("title", "")]
            if row.get("source"):
                parts.append(f"Source: {row['source']}")
            if row.get("date"):
                date_val = row["date"]
                if hasattr(date_val, "isoformat"):
                    parts.append(f"Date: {date_val.isoformat()}")
            if row.get("url"):
                parts.append(f"URL: {row['url']}")
            articles.append("\n".join(parts))

        return articles

    def to_dict(
        self,
        keyword: str = "",
        num_days: int = 7,
        num_pages: int = 2,
        limit: int = 15,
    ) -> dict:
        """Get news as dictionary for JSON serialization."""
        df = self.scrape(keyword, num_days, num_pages, limit)
        records = df.to_dict(orient="records")

        for r in records:
            if r.get("date") and hasattr(r["date"], "isoformat"):
                r["date"] = r["date"].isoformat()

        return {
            "keyword": keyword,
            "source": self.source,
            "news_articles": self.get_news_articles(keyword, num_days, num_pages, limit),
            "articles": records,
        }
