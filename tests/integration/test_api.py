import httpx
import pytest

from finrag_indic.api.app import create_app
from finrag_indic.config import Settings


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

