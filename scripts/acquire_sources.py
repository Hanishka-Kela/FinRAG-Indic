from argparse import ArgumentParser
from pathlib import Path

from finrag_indic.errors import AcquisitionError
from finrag_indic.knowledge.acquisition import SourceDownloader
from finrag_indic.knowledge.manifest import load_manifest, save_manifest
from finrag_indic.knowledge.models import ApprovalStatus, ExtractionStatus, SourceDocument


def main() -> None:
    parser = ArgumentParser(description="Download only explicitly approved source documents")
    parser.add_argument(
        "--manifest", type=Path, default=Path("configs/sources.json")
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/sources"))
    parser.add_argument(
        "--document-id", action="append", default=[], help="Limit acquisition to an ID"
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    downloader = SourceDownloader()
    approved = [
        source for source in manifest.sources if source.approval_status == ApprovalStatus.APPROVED
        and (not args.document_id or source.document_id in args.document_id)
    ]
    if not approved:
        raise SystemExit("No source URLs are explicitly approved; nothing was downloaded.")
    failures: list[str] = []
    for source in approved:
        try:
            updated = downloader.download(source, args.raw_root)
            print(
                f"Acquired {source.document_id}: status={updated.http_status} "
                f"content_type={updated.content_type} sha256={updated.sha256}"
            )
        except AcquisitionError as exc:
            updated = SourceDocument.model_validate(
                {
                    **source.model_dump(),
                    "http_status": exc.http_status,
                    "content_type": exc.content_type,
                    "resolved_source_url": exc.resolved_source_url,
                    "retrieval_timestamp": exc.retrieval_timestamp,
                    "sha256": exc.sha256,
                    "local_file_path": exc.local_file_path,
                    "extraction_status": (
                        ExtractionStatus.FAILED
                        if exc.local_file_path is not None
                        else ExtractionStatus.NOT_ACQUIRED
                    ),
                    "extraction_error": str(exc),
                }
            )
            failures.append(f"{source.document_id}: {exc}")
            print(f"FAILED {source.document_id}: {exc}")
        manifest.replace(updated)
        save_manifest(manifest, args.manifest)
    if failures:
        raise SystemExit("Source acquisition failures:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()

