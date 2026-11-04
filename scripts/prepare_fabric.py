from argparse import ArgumentParser
import hashlib
from datetime import UTC, datetime
from pathlib import Path

from finrag_indic.datasets.fabric import (
    FABRIC_REVISION,
    FabricSnapshot,
    acquire_fabric,
    load_fabric,
    save_snapshot,
    write_review_template,
)


def main() -> None:
    parser = ArgumentParser(description="Acquire and validate the pinned FABRIC dataset")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/datasets/fabric"),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Download the pinned artifact even when the local snapshot exists",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed/datasets/fabric"),
    )
    args = parser.parse_args()

    parquet_path = args.raw_root / FABRIC_REVISION / "train.parquet"
    if args.refresh or not parquet_path.exists():
        snapshot = acquire_fabric(parquet_path)
    else:
        content = parquet_path.read_bytes()
        snapshot = FabricSnapshot(
            retrieved_at=datetime.fromtimestamp(parquet_path.stat().st_mtime, UTC),
            file_sha256=hashlib.sha256(content).hexdigest(),
            record_count=len(load_fabric(parquet_path)),
        )
    records = load_fabric(parquet_path)
    snapshot = snapshot.model_copy(
        update={
            "missing_source_description_count": sum(
                not record.source_description.strip() for record in records
            )
        }
    )
    write_review_template(records, args.processed_root / "review_template.jsonl")
    save_snapshot(snapshot, args.processed_root / "snapshot.json")
    print(f"Validated {len(records)} unreviewed FABRIC records at revision {FABRIC_REVISION}")


if __name__ == "__main__":
    main()

