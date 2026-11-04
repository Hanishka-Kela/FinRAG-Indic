class FinRAGError(Exception):
    """Base class for expected application errors."""


class ManifestError(FinRAGError):
    pass


class AcquisitionError(FinRAGError):
    def __init__(
        self,
        message: str,
        *,
        http_status: int | None = None,
        content_type: str | None = None,
        resolved_source_url: str | None = None,
    ) -> None:
        super().__init__(message)
        self.http_status = http_status
        self.content_type = content_type
        self.resolved_source_url = resolved_source_url


class ExtractionError(FinRAGError):
    pass


class EmbeddingError(FinRAGError):
    pass


class IndexCompatibilityError(FinRAGError):
    pass


class EmptyCorpusError(FinRAGError):
    pass

