"""
Ingests PDF documents from the docs/ folder, splits them into chunks,
embeds them with a local HuggingFace model, and persists them to ChromaDB.

Run directly:
    python -m app.ingest docs/your_file.pdf
"""

import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def ingest_docs(pdf_path: str, persist_dir: str = PERSIST_DIR) -> int:
    """Load a PDF, chunk it, embed it, and persist to ChromaDB. Returns chunk count."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {pdf_path}")

    loader = PyPDFLoader(str(path))
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=persist_dir,
    )
    vectorstore.persist()

    print(f"Ingested {len(chunks)} chunks from {path.name} into {persist_dir}")
    return len(chunks)


def ingest_folder(folder: str = "docs", persist_dir: str = PERSIST_DIR) -> int:
    """Ingest every PDF in a folder. Returns total chunk count."""
    folder_path = Path(folder)
    pdfs = sorted(folder_path.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {folder_path.resolve()}")
        return 0

    total = 0
    for pdf in pdfs:
        total += ingest_docs(str(pdf), persist_dir=persist_dir)
    return total


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ingest_docs(sys.argv[1])
    else:
        ingest_folder()