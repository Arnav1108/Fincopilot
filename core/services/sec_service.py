#sec_service.py


import os
import requests
from sec_edgar_downloader import Downloader


DATA_DIR = "data/sec_filings"


# ---------------------------------------------------------
# Resolve Company → Ticker (Official SEC mapping)
# ---------------------------------------------------------
def _resolve_ticker(company: str) -> str:
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {"User-Agent": "FinCopilot (youremail@example.com)"}

    response = requests.get(url, headers=headers, timeout=10)
    data = response.json()

    company_lower = company.strip().lower()

    for entry in data.values():
        name = entry["title"].lower()
        ticker = entry["ticker"]

        if company_lower in name:
            return ticker

    raise ValueError(f"Could not resolve ticker for '{company}'")


# ---------------------------------------------------------
# Extract primary 10-K from full submission container
# ---------------------------------------------------------
def extract_primary_10k(submission_path: str) -> str:
    """
    Extract the <TYPE>10-K document from full-submission.txt
    and save it as primary_10k.html
    """

    with open(submission_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    documents = content.split("<DOCUMENT>")

    for doc in documents:
        if "<TYPE>10-K" in doc:

            start = doc.find("<TEXT>")
            end = doc.find("</TEXT>")

            if start == -1 or end == -1:
                continue

            extracted = doc[start + 6:end]

            output_path = submission_path.replace(
                "full-submission.txt",
                "primary_10k.html"
            )

            with open(output_path, "w", encoding="utf-8") as out:
                out.write(extracted)

            return output_path

    raise Exception("Primary 10-K document not found inside submission.")


# ---------------------------------------------------------
# Main Fetch Function
# ---------------------------------------------------------
def fetch_10k(company: str, year: str) -> dict:
    """
    Fetch and extract 10-K filing for a company + year.
    Returns metadata + path to cleaned primary document.
    """

    try:
        os.makedirs(DATA_DIR, exist_ok=True)

        ticker = _resolve_ticker(company)
        ticker_upper = ticker.upper()

        # Download filings
        dl = Downloader("FinCopilot", "youremail@example.com", DATA_DIR)
        dl.get("10-K", ticker_upper)

        filings_root = os.path.join(
            DATA_DIR,
            "sec-edgar-filings",
            ticker_upper,
            "10-K"
        )

        if not os.path.exists(filings_root):
            return {"status": "failed", "reason": "No filings found."}

        year_suffix = str(year)[-2:]  # 2023 → "23"

        selected_file = None
        filing_date = None

        # Find correct accession folder
        for filing_folder in os.listdir(filings_root):
            filing_path = os.path.join(filings_root, filing_folder)

            if not os.path.isdir(filing_path):
                continue

            # Accession format contains -YY-
            if f"-{year_suffix}-" not in filing_folder:
                continue

            # Locate full submission container
            submission_file = None
            for file in os.listdir(filing_path):
                if file.endswith(".txt"):
                    submission_file = os.path.join(filing_path, file)
                    break

            if not submission_file:
                continue

            # Extract actual 10-K document
            selected_file = extract_primary_10k(submission_file)
            filing_date = filing_folder
            break

        if not selected_file:
            return {
                "status": "failed",
                "reason": f"No 10-K found for year {year}"
            }

        # Preview
        with open(selected_file, "r", encoding="utf-8", errors="ignore") as f:
            preview = f.read(1000)

        return {
            "status": "success",
            "company": company,
            "year": year,
            "form_type": "10-K",
            "filing_date": filing_date,
            "local_path": selected_file,
            "preview": preview
        }

    except Exception as e:
        return {
            "status": "failed",
            "reason": str(e)
        }