import json
from argparse import ArgumentParser
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from finrag_indic.config import get_settings
from finrag_indic.evaluation.retrieval import BenchmarkQuestion
from finrag_indic.retrieval.embeddings import BgeM3Embedder


def main() -> None:
    parser = ArgumentParser(description="Generate unverified evidence candidates")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("configs/benchmarks/retrieval_v1.json"),
    )
    parser.add_argument(
        "--english-index", type=Path, default=Path("data/indexes/english")
    )
    parser.add_argument(
        "--multilingual-index",
        type=Path,
        default=Path("data/indexes/multilingual-genuine-v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/benchmarks/retrieval_v1_candidates.json"),
    )
    args = parser.parse_args()
    settings = get_settings()
    questions = [
        BenchmarkQuestion.model_validate(item)
        for item in json.loads(args.benchmark.read_text(encoding="utf-8"))
    ]
    flattened = [
        (question, language, variant)
        for question in questions
        for language, variant in question.variants.items()
    ]
    embedder = BgeM3Embedder(
        model_id=settings.embedding_model,
        revision=settings.embedding_revision,
        cache_folder=Path("data/models/sentence-transformers"),
        device=settings.embedding_device,
    )
    vectors = np.vstack(
        [
            embedder.embed([variant.question], batch_size=1)[0]
            for _, _, variant in flattened
        ]
    )
    from finrag_indic.retrieval.index import FaissRetriever

    indexes = {
        "english_baseline_v1": FaissRetriever.load(
            args.english_index, expected_embedding=embedder.metadata
        ),
        args.multilingual_index.name.replace("-", "_"): FaissRetriever.load(
            args.multilingual_index, expected_embedding=embedder.metadata
        ),
    }
    records = []
    for (question, language, variant), vector in zip(flattened, vectors):
        retrievals = {}
        for index_name, retriever in indexes.items():
            hits = retriever.search(vector, top_k=5)
            retrievals[index_name] = [
                {
                    "rank": hit.rank,
                    "score": hit.score,
                    "chunk_id": hit.chunk.chunk_id,
                    "document_id": hit.chunk.document_id,
                    "language": hit.chunk.language.value,
                    "page_number": hit.chunk.page_number,
                    "source_url": str(hit.chunk.source_url),
                    "passage_excerpt": hit.chunk.text[:700],
                    "review_flags": hit.chunk.review_flags,
                }
                for hit in hits
            ]
        records.append(
            {
                "canonical_id": question.canonical_id,
                "topic": question.topic,
                "query_language": language,
                "question": variant.question,
                "translation_status": variant.translation_status,
                "evidence_status": variant.evidence_status,
                "verified_gold_chunk_ids": variant.verified_gold_chunk_ids,
                "candidate_evidence_note": (
                    "Top-5 retrieval candidates for human review; none is gold evidence."
                ),
                "retrievals": retrievals,
            }
        )
    payload = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark_status": "candidate_evidence_unverified",
        "question_count": len(questions),
        "variant_count": len(records),
        "embedding": embedder.metadata.model_dump(),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(records)} unverified review records to {args.output}")


if __name__ == "__main__":
    main()
