#rag_tools.py

import os 
import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool
from core.rag.rag import search,list_indexes

load_dotenv()


@tool
def search_documents(query: str, index_name: str = None) -> str:
    """
    Search uploaded financial documents.
    Optionally restrict to a specific index.
    """

    try:
        results = search(query, index_name=index_name, k=20)

        if not results:
            return "No relevant documents found."

        formatted = []
        for r in results:
            citation = f"[{r['filename']}, p.{r['page']}]"
            formatted.append(f"{citation}\n{r['content']}")

        return "\n\n".join(formatted)

    except Exception as e:
        print("SEARCH ERROR:", str(e))
        return "⚠️ Document retrieval failed."