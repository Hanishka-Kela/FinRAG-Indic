import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request, Response, status

from finrag_indic.config import Settings, get_settings
from finrag_indic.contracts import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SystemStatusResponse,
)
from finrag_indic.logging import configure_logging
from finrag_indic.rag.service import ChatService, create_chat_service


def create_app(
    settings: Settings | None = None,
    *,
    chat_service: ChatService | None = None,
) -> FastAPI:
    app_settings = settings or get_settings()
    service = chat_service or create_chat_service(app_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        configure_logging(app_settings.log_level)
        logging.getLogger(__name__).info(
            "application_started",
            extra={"environment": app_settings.environment},
        )
        yield
        close = getattr(service.provider, "close", None)
        if close is not None:
            await close()
        logging.getLogger(__name__).info("application_stopped")

    application = FastAPI(
        title=app_settings.app_name,
        version=app_settings.app_version,
        description="Evidence-grounded multilingual financial literacy API.",
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def request_logging(request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = request.headers.get("X-Request-ID", str(uuid4()))
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        logging.getLogger("finrag_indic.http").info(
            "request_completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round((time.perf_counter() - started) * 1000, 3),
            },
        )
        return response

    @application.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        return HealthResponse(
            service=app_settings.app_name,
            status="ok",
            version=app_settings.app_version,
            environment=app_settings.environment,
        )

    @application.get(
        "/system/status", response_model=SystemStatusResponse, tags=["system"]
    )
    async def system_status() -> SystemStatusResponse:
        index_path = app_settings.retrieval_index_path
        index_available = all(
            (index_path / filename).is_file()
            for filename in ("vectors.faiss", "metadata.json", "chunks.jsonl")
        )
        llm_configured = app_settings.llm_provider == "mock" or bool(
            app_settings.llm_model and app_settings.llm_api_key
        )
        return SystemStatusResponse(
            status="ready" if index_available and llm_configured else "degraded",
            llm_provider=app_settings.llm_provider,
            llm_model=(
                app_settings.llm_model
                if app_settings.llm_provider == "openai"
                else "deterministic-mock-v1"
            ),
            llm_configured=llm_configured,
            retrieval_index=str(index_path),
            retrieval_index_available=index_available,
            embedding_model=(
                f"{app_settings.embedding_model}@{app_settings.embedding_revision}"
            ),
            supported_languages=["en", "hi", "ta"],
        )

    @application.post("/chat", response_model=ChatResponse, tags=["chat"])
    async def chat(payload: ChatRequest, response: Response) -> ChatResponse:
        result = await service.chat(payload)
        if result.error is not None:
            response.status_code = {
                "sensitive_financial_information": status.HTTP_400_BAD_REQUEST,
                "question_too_long": status.HTTP_413_CONTENT_TOO_LARGE,
                "provider_not_configured": status.HTTP_503_SERVICE_UNAVAILABLE,
                "retrieval_unavailable": status.HTTP_503_SERVICE_UNAVAILABLE,
                "provider_request_failed": status.HTTP_502_BAD_GATEWAY,
                "invalid_generated_citations": status.HTTP_502_BAD_GATEWAY,
            }.get(result.error.code, status.HTTP_200_OK)
        return result

    return application


app = create_app()

