from finrag_indic.contracts import ProviderRequest, ProviderResponse


class DeterministicMockProvider:
    """Fixed-response provider for tests; it performs no model inference."""

    def __init__(
        self,
        response_text: str | None = None,
        *,
        abstained: bool = False,
    ) -> None:
        self.response_text = response_text
        self.abstained = abstained

    def _response_text(self, request: ProviderRequest) -> str:
        if self.response_text is not None:
            return self.response_text
        localized = {
            "en": "This deterministic demonstration answer is grounded in source [S1].",
            "hi": "यह नियतात्मक प्रदर्शन उत्तर स्रोत [S1] पर आधारित है।",
            "ta": "இந்த நிர்ணயிக்கப்பட்ட செயல்விளக்கப் பதில் ஆதாரம் [S1] அடிப்படையிலானது.",
        }
        return localized[request.language]

    def invoke(self, request: ProviderRequest) -> ProviderResponse:
        return ProviderResponse(
            text=self._response_text(request),
            provider="mock",
            model="deterministic-mock-v1",
            abstained=self.abstained,
        )

    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        return self.invoke(request)

