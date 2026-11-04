from typing import Protocol

from finrag_indic.contracts import ProviderRequest, ProviderResponse


class TextProvider(Protocol):
    """Replaceable boundary for a future text-generation provider."""

    def invoke(self, request: ProviderRequest) -> ProviderResponse: ...

