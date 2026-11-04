from argparse import ArgumentParser
from pathlib import Path

from finrag_indic.errors import AcquisitionError
from finrag_indic.knowledge.acquisition import SourceDownloader
from finrag_indic.knowledge.manifest import load_manifest, save_manifest
from finrag_indic.knowledge.models import ApprovalStatus


def main() -> None:
    parser = ArgumentParser(description="Download only explicitly approved source documents")
    parser.add_argument(
        "--manifest", type=Path, default=Path("configs/sources.json")
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/sources"))
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    downloader = SourceDownloader()
    approved = [
        source for source in manifest.sources if source.approval_status == ApprovalStatus.APPROVED
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
            updated = source.model_copy(
                update={
                    "http_status": exc.http_status,
                    "content_type": exc.content_type,
                    "resolved_source_url": exc.resolved_source_url,
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

