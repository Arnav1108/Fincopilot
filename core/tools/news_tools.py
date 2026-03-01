#news_tools.py

import os 
import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool
from core.rag.rag import search,list_indexes

load_dotenv()

@tool
def fetch_news(query: str) -> str:
    """
    Fetch recent financial news articles for a company or topic.
    
    Args:
        query: The company name or topic to search for (e.g., "Apple", "Google", "Tesla earnings")
    
    Use this when the user asks about recent events, news, market updates,
    or anything that requires current information not found in documents.
    """

    try:
        api_key = os.getenv("NEWS_API_KEY","")

        if not api_key:
            return "⚠️ Live news unavailable (API key missing)."

        response = httpx.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": query,
                "apiKey": api_key,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 5
            },
            timeout=10
        )

        data = response.json()

        if data.get("status") != "ok":
            return "⚠️ News service error."

        articles = data.get("articles", [])

        if not articles:
            return "No recent news found."

        formatted = []
        for i, article in enumerate(articles, 1):
            formatted.append(
                f"[Article {i}] {article.get('title')} "
                f"({article.get('source', {}).get('name')})"
            )

        return "\n".join(formatted)

    except Exception as e:
        print("NEWS ERROR:", str(e))
        return "⚠️ Live news service unavailable."
