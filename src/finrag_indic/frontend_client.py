from typing import Any

import httpx


class BackendClientError(Exception):
    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class BackendClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            transport=transport,
        )

    def status(self) -> dict[str, Any]:
        try:
            response = self.client.get("/system/status")
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise BackendClientError("Backend status is unavailable", retryable=True) from exc

    def chat(self, *, question: str, language: str) -> dict[str, Any]:
        try:
            response = self.client.post(
                "/chat", json={"question": question, "language": language}
            )
            payload = response.json()
        except httpx.TimeoutException as exc:
            raise BackendClientError("The backend request timed out", retryable=True) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise BackendClientError("The backend could not be reached", retryable=True) from exc
        if response.is_error:
            error = payload.get("error") if isinstance(payload, dict) else None
            message = (
                error.get("message")
                if isinstance(error, dict)
                else f"Backend returned HTTP {response.status_code}"
            )
            retryable = bool(error.get("retryable")) if isinstance(error, dict) else False
            raise BackendClientError(message, retryable=retryable)
        return payload
