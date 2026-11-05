import hashlib
import json
from argparse import ArgumentParser
from pathlib import Path

from finrag_indic.config import get_settings
from finrag_indic.errors import ExtractionError
from finrag_indic.knowledge.ingestion import chunk_blocks, extract_document
from finrag_indic.knowledge.manifest import load_manifest, save_manifest
from finrag_indic.knowledge.models import Chunk, ExtractionStatus
from finrag_indic.retrieval.embeddings import BgeM3Embedder, embed_with_cache


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_verified_baseline(config_path: Path) -> list[Chunk]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = Path(config["index_path"])
    for filename, expected in config["files"].items():
        actual = sha256_file(root / filename)
        if actual != expected:
            raise SystemExit(f"Baseline integrity failure for {filename}: {actual}")
    chunks = [
        Chunk.model_validate_json(line)
        for line in (root / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(chunks) != config["vector_count"]:
        raise SystemExit("Baseline vector/chunk count does not match its lock file")
    return chunks


def main() -> None:
    parser = ArgumentParser(description="Build the versioned genuine multilingual index")
    parser.add_argument(
        "--baseline", type=Path, default=Path("configs/baselines/english_v1.json")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("configs/sources_extension.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/indexes/multilingual-genuine-v1")
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path("data/processed/embeddings/multilingual-genuine-v1.npz"),
    )
    args = parser.parse_args()
    settings = get_settings()
    baseline_chunks = load_verified_baseline(args.baseline)
    manifest = load_manifest(args.manifest)
    extension_chunks: list[Chunk] = []
    for source in manifest.sources:
        if source.extraction_status == ExtractionStatus.NOT_ACQUIRED:
            continue
        try:
            blocks = extract_document(source)
            chunks = chunk_blocks(
                source,
                blocks,
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap,
                min_chunk_tokens=settings.min_chunk_tokens,
            )
            extension_chunks.extend(chunks)
            manifest.replace(
                source.model_copy(
                    update={"extraction_status": ExtractionStatus.EXTRACTED,
                            "extraction_error": None}
                )
            )
        except ExtractionError as exc:
            manifest.replace(
                source.model_copy(
                    update={"extraction_status": ExtractionStatus.FAILED,
                            "extraction_error": str(exc)}
                )
            )
    save_manifest(manifest, args.manifest)
    all_chunks = baseline_chunks + extension_chunks
    embedder = BgeM3Embedder(
        model_id=settings.embedding_model,
        revision=settings.embedding_revision,
        cache_folder=Path("data/models/sentence-transformers"),
        device=settings.embedding_device,
    )
    vectors = embed_with_cache(
        embedder,
        [chunk.text for chunk in all_chunks],
        batch_size=settings.embedding_batch_size,
        cache_path=args.embedding_cache,
    )
    from finrag_indic.retrieval.index import FaissRetriever

    retriever = FaissRetriever.build(
        chunks=all_chunks,
        vectors=vectors,
        embedding=embedder.metadata,
        corpus_view="multilingual-genuine-v1",
    )
    retriever.save(args.output)
    counts: dict[str, int] = {}
    for chunk in all_chunks:
        counts[chunk.language.value] = counts.get(chunk.language.value, 0) + 1
    print(f"Built {args.output} with {len(all_chunks)} vectors; languages={counts}")


if __name__ == "__main__":
    main()
