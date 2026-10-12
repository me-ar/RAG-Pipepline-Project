"""
Choose your LLM backend. Two free options:

Option A - Groq (fast, hosted, free tier, no local GPU needed):
    1. Sign up at https://console.groq.com
    2. export GROQ_API_KEY=your_key   (or put it in .env)

Option B - Ollama (fully local, no API key, needs the Ollama app running):
    1. Install Ollama: https://ollama.com
    2. Run: ollama pull llama3

Set LLM_BACKEND=groq or LLM_BACKEND=ollama in your .env file to switch.
"""

import os
from dotenv import load_dotenv

load_dotenv()

LLM_BACKEND = os.getenv("LLM_BACKEND", "groq")


def get_llm():
    if LLM_BACKEND == "ollama":
        from langchain_community.llms import Ollama
        return Ollama(model="llama3")

    # default: groq
    from langchain_groq import ChatGroq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Add it to your .env file, or set "
            "LLM_BACKEND=ollama in .env to run fully local instead."
        )
    return ChatGroq(model="openai/gpt-oss-20b", temperature=0, groq_api_key=api_key)