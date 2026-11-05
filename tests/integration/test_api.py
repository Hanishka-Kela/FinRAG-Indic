import httpx
import pytest

from finrag_indic.api.app import create_app
from finrag_indic.config import Settings
from finrag_indic.contracts import (
    ChatError,
    ChatResponse,
    LatencyInfo,
)


class StubChatService:
    provider = None

    async def chat(self, payload):  # type: ignore[no-untyped-def]
        error = (
            ChatError(
                code="provider_request_failed",
                message="Temporary provider failure",
                retryable=True,
            )
            if payload.question == "fail"
            else None
        )
        return ChatResponse(
            answer="उत्तर [S1]" if payload.language == "hi" else "Answer [S1]",
            language=payload.language,
            evidence=[],
            citations=[],
            abstained=error is not None,
            provider="stub",
            model="stub-v1",
            latency=LatencyInfo(retrieval_ms=1, generation_ms=2, total_ms=3),
            error=error,
        )


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_health_endpoint() -> None:
    settings = Settings(environment="test", _env_file=None)
    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/health", headers={"X-Request-ID": "test-request"}
            )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "test-request"
    assert response.json() == {
        "service": "FinRAG-Indic",
        "status": "ok",
        "version": "0.1.0",
        "environment": "test",
    }


@pytest.mark.anyio
async def test_unknown_route_returns_not_found() -> None:
    settings = Settings(environment="test", _env_file=None)
    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/not-a-route")

    assert response.status_code == 404


@pytest.mark.anyio
async def test_chat_endpoint_accepts_manual_language_selection() -> None:
    settings = Settings(environment="test", _env_file=None)
    app = create_app(settings, chat_service=StubChatService())  # type: ignore[arg-type]
    transport = httpx.ASGITransport(app=app)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/chat", json={"question": "महंगाई क्या है?", "language": "hi"}
            )

    assert response.status_code == 200
    assert response.json()["language"] == "hi"
    assert response.json()["answer"] == "उत्तर [S1]"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "payload",
    [
        {"question": "", "language": "en"},
        {"question": "question", "language": "bn"},
        {"question": "x" * 2001, "language": "en"},
    ],
)
async def test_chat_request_validation(payload: dict[str, str]) -> None:
    settings = Settings(environment="test", _env_file=None)
    app = create_app(settings, chat_service=StubChatService())  # type: ignore[arg-type]
    transport = httpx.ASGITransport(app=app)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/chat", json=payload)

    assert response.status_code == 422


@pytest.mark.anyio
async def test_provider_error_maps_to_bad_gateway() -> None:
    settings = Settings(environment="test", _env_file=None)
    app = create_app(settings, chat_service=StubChatService())  # type: ignore[arg-type]
    transport = httpx.ASGITransport(app=app)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/chat", json={"question": "fail", "language": "en"}
            )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "provider_request_failed"


@pytest.mark.anyio
async def test_system_status_reports_missing_index_and_provider_configuration(
    tmp_path,
) -> None:
    settings = Settings(
        environment="test",
        llm_provider="openai",
        llm_model=None,
        llm_api_key=None,
        retrieval_index_path=tmp_path / "missing",
        _env_file=None,
    )
    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/system/status")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert not response.json()["llm_configured"]
    assert not response.json()["retrieval_index_available"]


@pytest.mark.anyio
async def test_chat_reports_missing_hosted_credentials_without_loading_index(
    tmp_path,
) -> None:
    settings = Settings(
        environment="test",
        llm_provider="openai",
        llm_model="account-specific-model",
        llm_api_key=None,
        retrieval_index_path=tmp_path / "missing",
        _env_file=None,
    )
    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/chat", json={"question": "What is inflation?", "language": "en"}
            )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "provider_not_configured"
    assert response.json()["evidence"] == []

