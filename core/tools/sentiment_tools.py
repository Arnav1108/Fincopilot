#sentiment_tools.py

import os 
import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool
from core.rag.rag import search,list_indexes

load_dotenv()

@tool
def analyze_sentiment(text: str) -> str:
    """
    Analyze the financial sentiments of a piece of text.
    Use this when the user wants to understand the market mood,
    tone of an earnings call , or sentiment of a news article.
    Returns sentimetn score , label , and key driving words.
    """

    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)

    compound = scores["compound"]

    if compound >= 0.05:
        label = "BULLISH 📈"
        interpretation = (
            "The text has a positive financial tone. "
            "Suggests optimism, growth, or favorable conditions."
        )
    elif compound <= -0.05:
        label = "BEARISH 📉"
        interpretation = (
            "The text has a negative financial tone. "
            "Suggests pessimism, decline, or unfavorable conditions."
        )
    else:
        label = "NEUTRAL ➡️"
        interpretation = (
            "The text has a neutral financial tone. "
            "No strong positive or negative signals detected."
        )

    return (
        f"Sentiment: {label}\n"
        f"Compound Score : {compound:+.3f} "
        f"(range -1.0 to +1.0)\n"
        f"Positive Score : {scores['pos']:.3f}\n"
        f"Negative Score : {scores['neg']:.3f}\n"
        f"Neutral Score  : {scores['neu']:.3f}\n\n"
        f"Interpretation: {interpretation}"
    )