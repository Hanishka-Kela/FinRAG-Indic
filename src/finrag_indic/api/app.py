import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request

from finrag_indic.config import Settings, get_settings
from finrag_indic.contracts import HealthResponse
from finrag_indic.logging import configure_logging


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        configure_logging(app_settings.log_level)
        logging.getLogger(__name__).info(
            "application_started",
            extra={"environment": app_settings.environment},
        )
        yield
        logging.getLogger(__name__).info("application_stopped")

    application = FastAPI(
        title=app_settings.app_name,
        version=app_settings.app_version,
        description="Foundation API for the FinRAG-Indic research application.",
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

    return application


app = create_app()

