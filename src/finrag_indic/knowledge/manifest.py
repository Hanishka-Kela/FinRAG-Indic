import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from finrag_indic.errors import ManifestError
from finrag_indic.knowledge.models import SourceDocument


class SourceManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    sources: list[SourceDocument] = Field(default_factory=list)

    @model_validator(mode="after")
    def reject_duplicates(self) -> "SourceManifest":
        document_ids = [source.document_id for source in self.sources]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("duplicate document_id in source manifest")
        identities = [
            (str(source.official_source_url), source.language.value)
            for source in self.sources
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("duplicate source URL and language in source manifest")
        local_paths = [
            str(source.local_file_path)
            for source in self.sources
            if source.local_file_path is not None
        ]
        if len(local_paths) != len(set(local_paths)):
            raise ValueError("duplicate local file path in source manifest")
        return self

    def by_id(self, document_id: str) -> SourceDocument:
        for source in self.sources:
            if source.document_id == document_id:
                return source
        raise ManifestError(f"unknown document_id: {document_id}")

    def replace(self, document: SourceDocument) -> None:
        for index, existing in enumerate(self.sources):
            if existing.document_id == document.document_id:
                updated = list(self.sources)
                updated[index] = document
                validated = SourceManifest(
                    schema_version=self.schema_version,
                    sources=updated,
                )
                self.sources = validated.sources
                return
        raise ManifestError(f"unknown document_id: {document.document_id}")


def load_manifest(path: Path) -> SourceManifest:
    if not path.exists():
        raise ManifestError(f"source manifest does not exist: {path}")
    try:
        return SourceManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestError(f"invalid source manifest {path}: {exc}") from exc


def save_manifest(manifest: SourceManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.model_dump(mode="json")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def manifest_from_sources(sources: Iterable[SourceDocument]) -> SourceManifest:
    return SourceManifest(sources=list(sources))

