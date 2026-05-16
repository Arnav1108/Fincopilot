#loader.py
from bs4 import BeautifulSoup
import re
import os
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader
)
from bs4 import BeautifulSoup


def load_file(file_path: str, metadata: dict) -> list[Document]:

    ext = os.path.splitext(file_path)[-1].lower()

    # -------------------------------
    # HTML Handling (SEC filings)
    # -------------------------------
    if ext in [".html", ".htm"]:

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_html = f.read()

        soup = BeautifulSoup(raw_html, "html.parser")

        # Remove scripts and styles
        for tag in soup(["script", "style"]):
            tag.decompose()

        # Remove inline XBRL tags (ix:*)
        for tag in soup.find_all():
            if tag.name and tag.name.startswith("ix:"):
                tag.decompose()

        # Extract visible text
        text = soup.get_text(separator="\n")

        # Clean whitespace
        text = "\n".join(
            line.strip()
            for line in text.splitlines()
            if line.strip()
        )

        # Defensive validation (real 10-K should be large)
        if not text or len(text) < 1000:
            raise ValueError(
                "Extracted HTML text too small — possible parsing failure."
            )

        documents = [Document(page_content=text)]

    # -------------------------------
    # Other supported formats
    # -------------------------------
    else:

        LOADER_MAP = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".docx": Docx2txtLoader,
            ".csv": CSVLoader,
        }

        if ext not in LOADER_MAP:
            raise ValueError(
                f"Unsupported file type: '{ext}'. "
                f"Supported: {list(LOADER_MAP.keys()) + ['.html']}"
            )

        loader_class = LOADER_MAP[ext]

        if ext == ".txt":
            loader = loader_class(file_path, encoding="utf-8")
        else:
            loader = loader_class(file_path)

        documents = loader.load()

        if not documents:
            raise ValueError("Loader returned empty document list.")

    # -------------------------------
    # Add metadata
    # -------------------------------
    for doc in documents:
        doc.metadata["filename"] = os.path.basename(file_path)
        doc.metadata["company"] = metadata.get("company", "Unknown")
        doc.metadata["doc_type"] = metadata.get("doc_type", "General")
        doc.metadata["year"] = metadata.get("year", "Unknown")

    print(f"Loaded: {os.path.basename(file_path)} ({len(documents)} pages)")

    return documents


def load_multiple_files(file_paths: list[str], metadata: dict) -> list[Document]:

    all_documents = []

    for file_path in file_paths:
        try:
            docs = load_file(file_path, metadata)
            all_documents.extend(docs)
        except Exception as e:
            print(f"\nFailed to load {file_path}")
            print("Error:", repr(e))

    print(f"\nTotal loaded: {len(all_documents)} pages from {len(file_paths)} file(s)")

    return all_documents