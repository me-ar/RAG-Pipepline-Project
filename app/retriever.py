"""
Builds a retrieval + generation chain over the persisted ChromaDB vector store.
Uses the modern LangChain (1.x) LCEL style instead of the deprecated
RetrievalQA class, which was removed in langchain.chains.
"""

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.ingest import PERSIST_DIR, EMBEDDING_MODEL

PROMPT_TEMPLATE = """You are a regulatory compliance assistant.
Use ONLY the context below to answer the question.
If the answer is not in the context, say "I cannot find this in the provided documents."
Always cite the source page if available.

Context: {context}

Question: {question}

Answer:"""


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class RagChain:
    """Small wrapper so main.py can keep calling chain.invoke({'query': ...})."""

    def __init__(self, retriever, prompt, llm):
        self.retriever = retriever
        self.prompt = prompt
        self.llm = llm
        self.generate = prompt | llm | StrOutputParser()

    def invoke(self, inputs: dict):
        question = inputs["query"]
        docs = self.retriever.invoke(question)
        context = format_docs(docs)
        answer = self.generate.invoke({"context": context, "question": question})
        return {"result": answer, "source_documents": docs}


def get_chain(llm, persist_dir: str = PERSIST_DIR, k: int = 4):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    return RagChain(retriever, prompt, llm)