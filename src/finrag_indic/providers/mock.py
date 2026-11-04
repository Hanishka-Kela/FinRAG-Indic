from finrag_indic.contracts import ProviderRequest, ProviderResponse


class DeterministicMockProvider:
    """Fixed-response provider for tests; it performs no model inference."""

    def __init__(self, response_text: str = "mock response") -> None:
        self.response_text = response_text

    def invoke(self, request: ProviderRequest) -> ProviderResponse:
        del request
        return ProviderResponse(
            text=self.response_text,
            provider="mock",
            model="deterministic-mock-v1",
        )

