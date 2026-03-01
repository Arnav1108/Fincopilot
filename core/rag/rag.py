#rag.py

import os 
from core.rag.src.loader import load_multiple_files
from core.rag.src.chunking import chunk_documents
from core.rag.src.vectorstore import (
    build_index,
    add_to_index,
    search_all,
    search_one,
    INDEXES_DIR,
)

def ingest(file_paths: list[str], metadata: dict) -> dict:
    try:
        company = metadata.get("company", "unknown").lower().replace(" ","_")
        doc_type = metadata.get("doc_type","general").lower().replace(" ","_")
        year = str(metadata.get("year","0000"))
        index_name = f"{company}_{doc_type}_{year}"

        print(f"\n Starting ingestion -> index '{index_name}'")

        # Load
        documents = load_multiple_files(file_paths, metadata)

        if not documents:
            return {
                "status": "failed",
                "reason": "No documents could be loaded.",
                "index_name": index_name
            }

        # Chunk
        try:
            chunks = chunk_documents(documents)
        except Exception as e:
            print("CHUNK ERROR:", str(e))
            return {
                "status": "failed",
                "reason": "Document chunking failed.",
                "index_name": index_name
            }

        # Build / Update Index
        index_path = os.path.join(INDEXES_DIR, index_name)

        try:
            if os.path.exists(os.path.join(index_path, "index.faiss")):
                add_to_index(chunks, index_name)
                action = "updated"
            else:
                build_index(chunks, index_name)
                action = "created"
        except Exception as e:
            print("FAISS ERROR:", str(e))
            return {
                "status": "failed",
                "reason": "Vector index creation failed.",
                "index_name": index_name
            }

        print(f"\n Done — index '{index_name}' {action}")

        return {
            "status": "success",
            "action": action,
            "index_name": index_name,
            "files": len(file_paths),
            "pages": len(documents),
            "chunks": len(chunks),
        }

    except Exception as e:
        print("INGEST ERROR:", str(e))
        return {
            "status": "failed",
            "reason": "Unexpected ingestion failure.",
            "index_name": "unknown"
        }


def keyword_boost(results, query):
    keywords = ["revenue", "net income", "cash flow", "total assets"]

    boosted = []

    for r in results:
        score = 0
        content = r["content"].lower()

        for kw in keywords:
            if kw in content:
                score += 1

        boosted.append((score, r))

    boosted.sort(reverse=True, key=lambda x: x[0])

    return [r for _, r in boosted]


def search(query: str, index_name: str = None, k: int = 5) -> list[dict]:
    if not query or not query.strip():
        return []

    if index_name:
        return search_one(query, index_name, k=k)

    # DO NOT merge everything blindly
    return search_all(query, k=k)
    
def list_indexes() -> list[str]:

    if not os.path.exists(INDEXES_DIR):
        return []
    
    return[
        folder for folder in os.listdir(INDEXES_DIR)
        if os.path.isdir(os.path.join(INDEXES_DIR,folder))
    ]