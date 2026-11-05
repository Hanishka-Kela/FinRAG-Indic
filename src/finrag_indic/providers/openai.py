import openai
from openai import AsyncOpenAI

from finrag_indic.config import Settings
from finrag_indic.contracts import ProviderRequest, ProviderResponse
from finrag_indic.errors import ProviderConfigurationError, ProviderRequestError


class OpenAIProvider:
    """Hosted provider using an OpenAI-compatible asynchronous chat API."""

    def __init__(self, settings: Settings) -> None:
        if not settings.llm_api_key:
            raise ProviderConfigurationError(
                "FINRAG_LLM_API_KEY is required when FINRAG_LLM_PROVIDER=openai"
            )
        if not settings.llm_model:
            raise ProviderConfigurationError(
                "FINRAG_LLM_MODEL must name a model available to the configured account"
            )
        self.model = settings.llm_model
        self.temperature = settings.llm_temperature
        self.max_output_tokens = settings.llm_max_output_tokens
        options: dict[str, object] = {
            "api_key": settings.llm_api_key,
            "timeout": settings.llm_timeout_seconds,
            "max_retries": settings.llm_max_retries,
        }
        if settings.llm_base_url is not None:
            options["base_url"] = str(settings.llm_base_url)
        self.client = AsyncOpenAI(**options)  # type: ignore[arg-type]

    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        parameters: dict[str, object] = {
            "model": self.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": self.max_output_tokens,
        }
        if self.temperature is not None:
            parameters["temperature"] = self.temperature
        try:
            response = await self.client.chat.completions.create(
                **parameters
            )  # type: ignore[arg-type]
        except openai.AuthenticationError as exc:
            raise ProviderConfigurationError(
                "The hosted LLM rejected the configured credential"
            ) from exc
        except (openai.APITimeoutError, openai.RateLimitError) as exc:
            raise ProviderRequestError(
                "The hosted LLM is temporarily unavailable", retryable=True
            ) from exc
        except openai.APIConnectionError as exc:
            raise ProviderRequestError(
                "The hosted LLM could not be reached", retryable=True
            ) from exc
        except openai.APIStatusError as exc:
            retryable = exc.status_code in {408, 409, 429} or exc.status_code >= 500
            raise ProviderRequestError(
                f"The hosted LLM returned HTTP {exc.status_code}",
                retryable=retryable,
            ) from exc
        except openai.APIError as exc:
            raise ProviderRequestError("The hosted LLM request failed") from exc
        choices = getattr(response, "choices", None)
        output_text = None
        if choices:
            message = getattr(choices[0], "message", None)
            output_text = getattr(message, "content", None)
        if not isinstance(output_text, str):
            raise ProviderRequestError("The hosted LLM returned no text")
        output_text = output_text.strip()
        if not output_text:
            raise ProviderRequestError("The hosted LLM returned no text")
        marker = "INSUFFICIENT_EVIDENCE:"
        abstained = output_text.startswith(marker)
        if abstained:
            output_text = output_text.removeprefix(marker).strip()
        return ProviderResponse(
            text=output_text,
            provider="openai",
            model=self.model,
            abstained=abstained,
            request_id=getattr(response, "_request_id", None),
        )

    async def close(self) -> None:
        await self.client.close()
