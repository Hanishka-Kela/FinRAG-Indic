from functools import lru_cache
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

    llm_provider: str = "mock"
    llm_base_url: HttpUrl | None = None
    llm_model: str | None = None
    llm_api_key: str | None = Field(default=None, repr=False)

    @field_validator("llm_model", "llm_api_key", mode="before")
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

