from finrag_indic.contracts import ProviderRequest
from finrag_indic.providers.mock import DeterministicMockProvider


def test_mock_provider_is_deterministic() -> None:
    provider = DeterministicMockProvider(response_text="fixed")
    request = ProviderRequest(prompt="ignored", language="ta")

    first = provider.invoke(request)
    second = provider.invoke(request)

    assert first == second
    assert first.text == "fixed"
    assert first.model == "deterministic-mock-v1"

