from types import SimpleNamespace

import httpx
import openai
import pytest

from finrag_indic.config import Settings
from finrag_indic.contracts import ProviderRequest
from finrag_indic.errors import ProviderConfigurationError, ProviderRequestError
from finrag_indic.providers.openai import OpenAIProvider


def test_openai_provider_requires_credentials() -> None:
    settings = Settings(
        llm_provider="openai",
        llm_model="account-specific-model",
        llm_api_key=None,
        _env_file=None,
    )

    with pytest.raises(ProviderConfigurationError, match="FINRAG_LLM_API_KEY"):
        OpenAIProvider(settings)


def test_openai_provider_requires_explicit_model() -> None:
    settings = Settings(
        llm_provider="openai",
        llm_model=None,
        llm_api_key="test-secret",
        _env_file=None,
    )

    with pytest.raises(ProviderConfigurationError, match="FINRAG_LLM_MODEL"):
        OpenAIProvider(settings)


class FakeResponses:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.parameters: dict[str, object] | None = None

    async def create(self, **parameters):  # type: ignore[no-untyped-def]
        self.parameters = parameters
        if self.error:
            raise self.error
        return SimpleNamespace(
            output_text="Grounded answer [S1]", _request_id="req-test"
        )


@pytest.mark.anyio
async def test_openai_provider_uses_async_responses_and_generation_settings() -> None:
    settings = Settings(
        llm_provider="openai",
        llm_model="account-specific-model",
        llm_api_key="test-secret",
        llm_temperature=0.1,
        llm_max_output_tokens=321,
        llm_timeout_seconds=7,
        llm_max_retries=4,
        _env_file=None,
    )
    provider = OpenAIProvider(settings)
    assert provider.client.timeout == 7
    assert provider.client.max_retries == 4
    fake_responses = FakeResponses()
    original_client = provider.client
    provider.client = SimpleNamespace(responses=fake_responses)  # type: ignore[assignment]
    await original_client.close()

    result = await provider.generate(
        ProviderRequest(prompt="evidence prompt", language="en")
    )

    assert result.text == "Grounded answer [S1]"
    assert result.request_id == "req-test"
    assert fake_responses.parameters == {
        "model": "account-specific-model",
        "input": "evidence prompt",
        "max_output_tokens": 321,
        "temperature": 0.1,
    }


@pytest.mark.anyio
async def test_openai_timeout_becomes_retryable_provider_error() -> None:
    settings = Settings(
        llm_provider="openai",
        llm_model="account-specific-model",
        llm_api_key="test-secret",
        _env_file=None,
    )
    provider = OpenAIProvider(settings)
    original_client = provider.client
    provider.client = SimpleNamespace(  # type: ignore[assignment]
        responses=FakeResponses(
            error=openai.APITimeoutError(httpx.Request("POST", "https://api.test"))
        )
    )
    await original_client.close()

    with pytest.raises(ProviderRequestError) as error:
        await provider.generate(ProviderRequest(prompt="prompt", language="en"))

    assert error.value.retryable


@pytest.mark.anyio
@pytest.mark.parametrize("output_text", [None, 42, ""])
async def test_openai_provider_rejects_missing_output_text(output_text) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(
        llm_provider="openai",
        llm_model="account-specific-model",
        llm_api_key="test-secret",
        _env_file=None,
    )
    provider = OpenAIProvider(settings)
    original_client = provider.client
    provider.client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **parameters: _response_with_output(output_text)
        )
    )  # type: ignore[assignment]
    await original_client.close()

    with pytest.raises(ProviderRequestError, match="returned no text"):
        await provider.generate(ProviderRequest(prompt="prompt", language="en"))


async def _response_with_output(output_text):  # type: ignore[no-untyped-def]
    return SimpleNamespace(output_text=output_text, _request_id="req-test")
