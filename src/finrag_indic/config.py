from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, HttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and an optional .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="FINRAG_",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "FinRAG-Indic"
    app_version: str = "0.1.0"
    environment: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    frontend_api_url: HttpUrl = HttpUrl("http://127.0.0.1:8000")

    llm_provider: Literal["mock", "openai"] = "mock"
    llm_base_url: HttpUrl | None = None
    llm_model: str | None = None
    llm_api_key: str | None = Field(default=None, repr=False)
    llm_temperature: float | None = Field(default=0.2, ge=0, le=2)
    llm_max_output_tokens: int = Field(default=600, ge=64, le=4096)
    llm_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    llm_max_retries: int = Field(default=2, ge=0, le=10)

    embedding_model: str = "BAAI/bge-m3"
    embedding_revision: str = "5617a9f61b028005a4858fdac845db406aefb181"
    embedding_device: Literal["cpu", "mps", "cuda"] = "cpu"
    embedding_batch_size: int = Field(default=8, ge=1, le=256)
    chunk_size: int = Field(default=450, ge=50, le=2000)
    chunk_overlap: int = Field(default=75, ge=0, le=500)
    min_chunk_tokens: int = Field(default=20, ge=1, le=500)
    retrieval_top_k: int = Field(default=5, ge=1, le=100)
    retrieval_index_path: Path = Path("data/indexes/multilingual-genuine-v2")
    chat_max_question_chars: int = Field(default=2000, ge=100, le=2000)
    frontend_request_timeout_seconds: float = Field(default=90.0, gt=0, le=600)

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_must_be_smaller_than_chunk(cls, value: int, info) -> int:  # type: ignore[no-untyped-def]
        chunk_size = info.data.get("chunk_size", 450)
        if value >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return value

    @field_validator("min_chunk_tokens")
    @classmethod
    def minimum_chunk_must_fit(cls, value: int, info) -> int:  # type: ignore[no-untyped-def]
        chunk_size = info.data.get("chunk_size", 450)
        if value > chunk_size:
            raise ValueError("min_chunk_tokens cannot exceed chunk_size")
        return value

    @field_validator("llm_model", "llm_api_key", "llm_temperature", mode="before")
    @classmethod
    def empty_string_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("llm_base_url", mode="before")
    @classmethod
    def empty_url_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()

