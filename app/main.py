from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.retriever import get_chain
from app.llm import get_llm

app = FastAPI(title="RegQueryBot", description="Regulatory Document Q&A API")

_llm = None
_chain = None


def get_chain_lazy():
    """Build the chain on first use so the API can start even before docs are ingested."""
    global _llm, _chain
    if _chain is None:
        _llm = get_llm()
        _chain = get_chain(_llm)
    return _chain


class Query(BaseModel):
    question: str


@app.post("/query")
def query_docs(q: Query):
    try:
        chain = get_chain_lazy()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    result = chain.invoke({"query": q.question})
    sources = sorted({
        doc.metadata.get("source", "unknown")
        for doc in result["source_documents"]
    })
    return {
        "answer": result["result"],
        "sources": sources,
    }


@app.get("/health")
def health():
    return {"status": "ok"}