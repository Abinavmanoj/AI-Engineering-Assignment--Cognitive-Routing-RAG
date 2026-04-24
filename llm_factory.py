"""
llm_factory.py
--------------
Centralised factory that returns a LangChain chat model based on the
LLM_PROVIDER environment variable.  Supports Groq, OpenAI, and Ollama.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def get_llm(temperature: float = 0.7):
    """
    Return a LangChain ChatModel based on the LLM_PROVIDER env var.

    Supported providers
    -------------------
    groq   – Groq Cloud (llama3-8b-8192 by default, very fast & free tier)
    openai – OpenAI ChatCompletion (gpt-4o-mini by default)
    ollama – Local Ollama server (llama3 by default)
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model=os.getenv("GROQ_MODEL", "llama3-8b-8192"),
            temperature=temperature,
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=temperature,
        )

    elif provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "llama3"),
            temperature=temperature,
        )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER='{provider}'. "
            "Choose 'groq', 'openai', or 'ollama'."
        )
