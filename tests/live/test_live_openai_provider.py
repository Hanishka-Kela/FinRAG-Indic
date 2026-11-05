import os

import pytest

from finrag_indic.config import get_settings
from finrag_indic.contracts import ProviderRequest
from finrag_indic.providers.openai import OpenAIProvider


@pytest.mark.live
@pytest.mark.anyio
async def test_configured_hosted_provider_returns_text() -> None:
    if os.getenv("FINRAG_RUN_LIVE_LLM") != "1":
        pytest.skip("set FINRAG_RUN_LIVE_LLM=1 with provider credentials and model")
    settings = get_settings()
    provider = OpenAIProvider(settings)
    try:
        response = await provider.generate(
            ProviderRequest(
                prompt="Reply with the single word OK.",
                language="en",
            )
        )
    finally:
        await provider.close()

    assert response.text
    assert response.provider == "openai"
