import json
from argparse import ArgumentParser
from pathlib import Path

from finrag_indic.config import get_settings
from finrag_indic.knowledge.ingestion import (
    chunk_blocks,
    extract_document,
    extract_pdf_devanagari_glyphs,
    extract_rbi_failed_transaction_tables,
)
from finrag_indic.knowledge.manifest import load_manifest
from finrag_indic.knowledge.models import Chunk, ExtractionStatus
from finrag_indic.retrieval.embeddings import BgeM3Embedder, embed_with_cache

from build_multilingual_extension import load_verified_baseline


def main() -> None:
    parser = ArgumentParser(description="Build the Phase 3.5 quality index")
    parser.add_argument(
        "--baseline", type=Path, default=Path("configs/baselines/english_v1.json")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("configs/sources_extension.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/indexes/multilingual-genuine-v2")
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path("data/processed/embeddings/multilingual-genuine-v2.npz"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/processed/phase3_5_extraction_report.json"),
    )
    parser.add_argument(
        "--chunks-output",
        type=Path,
        default=Path("data/processed/multilingual-genuine-v2.chunks.jsonl"),
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    baseline_chunks = load_verified_baseline(args.baseline)
    manifest = load_manifest(args.manifest)
    extension_chunks: list[Chunk] = []
    report: dict[str, object] = {
        "index_version": "multilingual-genuine-v2",
        "baseline_chunk_count": len(baseline_chunks),
        "documents": {},
    }
    for source in manifest.sources:
        if source.extraction_status in {ExtractionStatus.NOT_ACQUIRED, ExtractionStatus.FAILED}:
            continue
        preserve_sections = False
        extraction_method = "pymupdf_standard"
        if source.document_id == "sebi-financial-education-hindi":
            blocks = extract_pdf_devanagari_glyphs(source)
            extraction_method = "embedded_cff_glyph_names_v1"
        elif source.document_id == "rbi-failed-transactions-tat-2019":
            blocks = extract_rbi_failed_transaction_tables(source)
            extraction_method = "pymupdf_table_rows_v1"
            preserve_sections = True
        else:
            blocks = extract_document(source)
        chunks = chunk_blocks(
            source,
            blocks,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
            min_chunk_tokens=settings.min_chunk_tokens,
            preserve_section_boundaries=preserve_sections,
            extraction_method=extraction_method,
        )
        extension_chunks.extend(chunks)
        combined = "\n".join(block.text for block in blocks)
        report["documents"][source.document_id] = {
            "language": source.language.value,
            "extraction_method": extraction_method,
            "block_count": len(blocks),
            "chunk_count": len(chunks),
            "replacement_character_count": combined.count("\ufffd"),
            "control_character_count": sum(
                1 for char in combined if ord(char) < 32 and char not in "\n\t"
            ),
        }
    all_chunks = baseline_chunks + extension_chunks
    args.chunks_output.parent.mkdir(parents=True, exist_ok=True)
    args.chunks_output.write_text(
        "".join(chunk.model_dump_json() + "\n" for chunk in all_chunks),
        encoding="utf-8",
    )
    counts: dict[str, int] = {}
    for chunk in all_chunks:
        counts[chunk.language.value] = counts.get(chunk.language.value, 0) + 1
    report["vector_count"] = len(all_chunks)
    report["language_chunk_counts"] = counts
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.prepare_only:
        print(f"Prepared {len(all_chunks)} chunks; languages={counts}")
        return
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
        corpus_view="multilingual-genuine-v2",
    )
    retriever.save(args.output)
    report["embedding"] = embedder.metadata.model_dump(mode="json")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Built {args.output} with {len(all_chunks)} vectors; languages={counts}")


if __name__ == "__main__":
    main()
