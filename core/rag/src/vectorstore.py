#vectorstore.py

import os 
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import re

load_dotenv()

INDEXES_DIR = os.getenv("FAISS_INDEXES_DIR","data/faiss_indexes")


def get_embeddings()-> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY",""),
    )

def build_index(chunks: list[Document], index_name: str) -> FAISS:
    embeddings = get_embeddings()
    index_path = os.path.join(INDEXES_DIR, index_name)
    os.makedirs(index_path, exist_ok=True)

    print(f"Building index '{index_name}' from {len(chunks)} chunks...")

    # Batch embedding
    batch_size = 1000
    vectorstore = None

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        print(f"Embedding batch {i} to {i+len(batch)}")

        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)

    vectorstore.save_local(index_path)
    print(f"Saved to {index_path}")

    return vectorstore

def add_to_index(chunks: list[Document],index_name: str)-> FAISS:

    embeddings = get_embeddings()
    index_path = os.path.join(INDEXES_DIR,index_name)

    print(f" Adding {len(chunks)} chunks to '{index_name}'...")

    vectorstore = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True    
    )

    vectorstore.add_documents(chunks)
    vectorstore.save_local(index_path)

    print(f"  Index '{index_name}' updated")
    return vectorstore




def search_all(query: str,k:int=5) -> list[dict]:
    
    if not os.path.exists(INDEXES_DIR):
        return []

    index_names = [
        folder for folder in os.listdir(INDEXES_DIR)
        if os.path.isdir(os.path.join(INDEXES_DIR,folder))

    ]

    if not index_names:
        return []
    
    embeddings = get_embeddings()
    merged = None
    for name in index_names:
        path = os.path.join(INDEXES_DIR,name)
        vs = FAISS.load_local(
            path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        if merged is None:
            merged = vs
        else:
            merged.merge_from(vs)

    results = merged.similarity_search_with_score(query,k=k)
    return _format_results(results)



def search_one(query: str, index_name: str, k: int = 5) -> list[dict]:

    index_path = os.path.join(INDEXES_DIR, index_name)

    if not os.path.exists(os.path.join(index_path, "index.faiss")):
        return []

    embeddings  = get_embeddings()
    vectorstore = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    results = vectorstore.similarity_search_with_score(query, k=k)
    return _format_results(results)

def _format_results(results: list[tuple]) -> list[dict]:

    formatted = []
    for doc, score in results:

        formatted.append({
            "content":   doc.page_content,
            "filename":  doc.metadata.get("filename",  "Unknown"),
            "company":   doc.metadata.get("company",   "Unknown"),
            "doc_type":  doc.metadata.get("doc_type",  "Unknown"),
            "year":      doc.metadata.get("year",      "Unknown"),
            "page":      doc.metadata.get("page",      "N/A"),
            "relevance": round(1 / (1 + score), 4),
        })

    return formatted