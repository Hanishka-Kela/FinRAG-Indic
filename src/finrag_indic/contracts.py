from typing import Literal

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    service: str
    status: Literal["ok"]
    version: str
    environment: str


class ProviderRequest(BaseModel):
    """Minimal provider contract used by deterministic Phase 2 test doubles."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    language: Literal["en", "hi", "ta"] = "en"


class ProviderResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    provider: str
    model: str

