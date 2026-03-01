#__inint__.py

from core.tools.rag_tools import search_documents
from core.tools.news_tools import fetch_news
from core.tools.sentiment_tools import analyze_sentiment
from core.tools.finance_tools import calculate_metrics
from core.tools.sec_fetch import fetch_sec_10k

TOOLS = [
    search_documents,
    fetch_news,
    analyze_sentiment,
    calculate_metrics,
    fetch_sec_10k
]