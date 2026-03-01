# chunking.py

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(documents: list[Document]) -> list[Document]:

    splitters = RecursiveCharacterTextSplitter(
        chunk_size = 1200,
        chunk_overlap=200,
        separators=["\n\n", "\n",". "," "]

    )

    chunks = splitters.split_documents(documents)

    print(f"  Chunking: {len(documents)} pages → {len(chunks)} chunks")

    return chunks