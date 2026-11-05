from typing import Protocol

from finrag_indic.contracts import ProviderRequest, ProviderResponse


class TextProvider(Protocol):
    """Replaceable asynchronous boundary for grounded text generation."""

    async def generate(self, request: ProviderRequest) -> ProviderResponse: ...

