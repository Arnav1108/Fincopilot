#sec_fetch.py

from langchain_core.tools import tool
from core.services.sec_service import fetch_10k


@tool
def fetch_sec_10k(company: str, year: str) -> dict:
    """
    Fetch 10-K filing for a given company and year from SEC.
    Returns structured result.
    """

    result = fetch_10k(company, year)

    if result["status"] == "failed":
        return {
            "status": "failed",
            "message": result["reason"]
        }

    return {
        "status": "success",
        "company": result["company"],
        "year": result["year"],
        "filing_date": result["filing_date"],
        "local_path": result["local_path"],
        "preview": result["preview"][:800]
    }