#test_rag.py

"""
tests/test_rag.py
End to end test — ingest a real PDF and search it.
"""

import sys
import os

# Make sure Python can find the core module
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.rag.rag import ingest, search, list_indexes

# ── Change this to your actual PDF filename ────────────────────────────────
FILE_NAME = "apple_report.pdf"  # <- change this
# ──────────────────────────────────────────────────────────────────────────

file_path = os.path.join("data", "uploads", FILE_NAME)

print("=" * 50)
print("STEP 1: Ingesting document...")
print("=" * 50)

result = ingest(
    file_paths=[file_path],
    metadata={
        "company":  "Test Company",
        "doc_type": "annual_report",
        "year":     2023,
    }
)

print("\nIngestion result:")
print(result)

print("\n" + "=" * 50)
print("STEP 2: Available indexes...")
print("=" * 50)
print(list_indexes())

print("\n" + "=" * 50)
print("STEP 3: Searching...")
print("=" * 50)

results = search("what is this document about", k=3)

for i, r in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(f"  Company  : {r['company']}")
    print(f"  File     : {r['filename']}")
    print(f"  Page     : {r['page']}")
    print(f"  Relevance: {r['relevance']}")
    print(f"  Content  : {r['content'][:200]}...")