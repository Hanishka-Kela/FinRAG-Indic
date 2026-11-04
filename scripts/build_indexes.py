import json
from argparse import ArgumentParser
from datetime import UTC, datetime
from pathlib import Path

from finrag_indic.config import get_settings
from finrag_indic.errors import ExtractionError
from finrag_indic.knowledge.ingestion import chunk_blocks, extract_document
from finrag_indic.knowledge.manifest import load_manifest, save_manifest
from finrag_indic.knowledge.models import ExtractionStatus
from finrag_indic.retrieval.embeddings import (
    BgeM3Embedder,
    benchmark_embeddings,
    embed_with_cache,
)


def main() -> None:
    parser = ArgumentParser(description="Extract approved sources and build FAISS indexes")
    parser.add_argument("--manifest", type=Path, default=Path("configs/sources.json"))
    parser.add_argument("--index-root", type=Path, default=Path("data/indexes"))
    parser.add_argument(
        "--embedding-cache", type=Path, default=Path("data/processed/embeddings")
    )
    parser.add_argument(
        "--benchmark-output",
        type=Path,
        default=Path("results/raw/embedding_benchmark.json"),
    )
    args = parser.parse_args()
    settings = get_settings()
    manifest = load_manifest(args.manifest)

    all_chunks = []
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
            all_chunks.extend(chunks)
            manifest.replace(
                source.model_copy(
                    update={
                        "extraction_status": ExtractionStatus.EXTRACTED,
                        "extraction_error": None,
                    }
                )
            )
        except ExtractionError as exc:
            manifest.replace(
                source.model_copy(
                    update={
                        "extraction_status": ExtractionStatus.FAILED,
                        "extraction_error": str(exc),
                    }
                )
            )
    save_manifest(manifest, args.manifest)
    if not all_chunks:
        raise SystemExit("No acquired source documents produced chunks; no index was built.")

    embedder = BgeM3Embedder(
        model_id=settings.embedding_model,
        revision=settings.embedding_revision,
        cache_folder=Path("data/models/sentence-transformers"),
        device=settings.embedding_device,
    )
    benchmark_sample = [chunk.text for chunk in all_chunks[: min(8, len(all_chunks))]]
    _, benchmark = benchmark_embeddings(
        embedder, benchmark_sample, batch_size=settings.embedding_batch_size
    )
    from finrag_indic.retrieval.index import FaissRetriever, corpus_view

    args.benchmark_output.parent.mkdir(parents=True, exist_ok=True)
    args.benchmark_output.write_text(
        json.dumps(
            {
                **benchmark.model_dump(mode="json"),
                "recorded_at": datetime.now(UTC).isoformat(),
                "note": "RSS delta is an observed process-memory delta, not hardware-wide peak memory.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    for name, chunks in (
        ("english", corpus_view(all_chunks, multilingual=False)),
        ("multilingual", corpus_view(all_chunks, multilingual=True)),
    ):
        if not chunks:
            continue
        cache_path = args.embedding_cache / f"{name}.npz"
        vectors = embed_with_cache(
            embedder,
            [chunk.text for chunk in chunks],
            batch_size=settings.embedding_batch_size,
            cache_path=cache_path,
        )
        retriever = FaissRetriever.build(
            chunks=chunks,
            vectors=vectors,
            embedding=embedder.metadata,
            corpus_view=name,
        )
        retriever.save(args.index_root / name)
        print(f"Built {name} IndexFlatIP with {len(chunks)} vectors")


if __name__ == "__main__":
    main()

