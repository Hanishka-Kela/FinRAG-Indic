import httpx
import pytest

from finrag_indic.frontend_client import BackendClient, BackendClientError


def test_frontend_client_sends_selected_language() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/chat"
        assert request.read() == '{"question":"प्रश्न","language":"hi"}'.encode()
        return httpx.Response(
            200,
            json={"answer": "उत्तर", "abstained": False, "citations": []},
        )

    client = BackendClient(
        "http://test", timeout=1, transport=httpx.MockTransport(handler)
    )

    assert client.chat(question="प्रश्न", language="hi")["answer"] == "उत्तर"


def test_frontend_client_surfaces_structured_backend_error() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            503,
            json={
                "error": {
                    "message": "Provider is not configured",
                    "retryable": False,
                }
            },
        )
    )
    client = BackendClient("http://test", timeout=1, transport=transport)

    with pytest.raises(BackendClientError, match="not configured"):
        client.chat(question="question", language="en")
