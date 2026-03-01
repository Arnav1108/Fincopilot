#router.py

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage,HumanMessage
import re

INTENT_PROMPT = """
You are a intent classifier for a financial AI system.

classify the user query into ONE of the following categories:


DOCUMENT   → Questions about uploaded financial documents.
NEWS       → Questions about latest news, recent events, updates.
CALCULATION→ User provides raw numbers and wants metrics.
GENERAL    → Finance theory or general knowledge.
COMPLEX

COMPLEX applies when the query requires:
- Combining documents and news
- Cause-effect analysis
- Strategic evaluation
- Multi-step reasoning

Return ONLY one word from:
DOCUMENT, NEWS, CALCULATION, GENERAL , COMPLEX
 """


def classify_intent(user_input: str) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    response = llm.invoke([
        SystemMessage(content="""
You are an intent classifier for a financial AI system.

Return ONLY one word from:

DOCUMENT
NEWS
CALCULATION
GENERAL
COMPLEX
FETCH_SEC
CONFIRM_SEC
CANCEL_SEC

Strict Rules:

1. If user says "fetch", "download", "get filing", "retrieve from SEC" → FETCH_SEC

2. If user says exactly "yes", "confirm", "ingest" → CONFIRM_SEC

3. If user says exactly "no", "cancel" → CANCEL_SEC

4. If query requires multi-step reasoning or combining sources → COMPLEX                      

5. Mentioning "10-K" alone does NOT mean fetch.
                      
Do NOT explain.
Return exactly one word.
"""),
        HumanMessage(content=user_input)
    ])

    raw = response.content.strip().upper()

    # Extract only the first valid keyword using regex
    match = re.search(r"(DOCUMENT|NEWS|CALCULATION|GENERAL|COMPLEX|FETCH_SEC|CONFIRM_SEC|CANCEL_SEC)", raw)

    if match:
        return match.group(1)

    return "GENERAL"