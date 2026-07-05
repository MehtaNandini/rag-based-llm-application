import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Application settings using Pydantic BaseSettings."""

    # Provider: mock, openai, ollama
    llm_provider: str = "mock"
    
    # OpenAI Settings
    openai_api_key: Optional[str] = None
    llm_model: str = "gpt-3.5-turbo"
    
    # Groq Settings
    groq_api_key: Optional[str] = None
    groq_model: str = "llama3-8b-8192"
    
    # Ollama Settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    
    # Embedding settings
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # ChromaDB settings
    chroma_persist_dir: str = "./chroma_db"
    
    # Retrieval settings
    top_k_results: int = 4
    
    # Chunking settings
    chunk_size: int = 1000
    chunk_overlap: int = 200

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
