import json
from argparse import ArgumentParser
from pathlib import Path

from finrag_indic.config import get_settings
from finrag_indic.evaluation.retrieval import BenchmarkQuestion, calculate_recalls
from finrag_indic.retrieval.embeddings import BgeM3Embedder


def main() -> None:
    parser = ArgumentParser(description="Evaluate only human-verified gold mappings")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("configs/benchmarks/retrieval_v1.json"),
    )
    parser.add_argument(
        "--index", type=Path, default=Path("data/indexes/multilingual-genuine-v1")
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    questions = [
        BenchmarkQuestion.model_validate(item)
        for item in json.loads(args.benchmark.read_text(encoding="utf-8"))
    ]
    verified = [
        (question.canonical_id, language, variant)
        for question in questions
        for language, variant in question.variants.items()
        if variant.evidence_status == "human_verified"
    ]
    evaluated: list[tuple[set[str], list[str]]] = []
    details = []
    if verified:
        settings = get_settings()
        embedder = BgeM3Embedder(
            model_id=settings.embedding_model,
            revision=settings.embedding_revision,
            cache_folder=Path("data/models/sentence-transformers"),
            device=settings.embedding_device,
        )
        vectors = embedder.embed(
            [variant.question for _, _, variant in verified],
            batch_size=settings.embedding_batch_size,
        )
        from finrag_indic.retrieval.index import FaissRetriever

        retriever = FaissRetriever.load(args.index, expected_embedding=embedder.metadata)
        for (canonical_id, language, variant), vector in zip(verified, vectors):
            retrieved = [
                hit.chunk.chunk_id for hit in retriever.search(vector, top_k=5)
            ]
            gold = set(variant.verified_gold_chunk_ids)
            evaluated.append((gold, retrieved))
            details.append(
                {"canonical_id": canonical_id, "language": language,
                 "gold_chunk_ids": sorted(gold), "retrieved_chunk_ids": retrieved}
            )
    payload = {
        "evaluated_variant_count": len(evaluated),
        "excluded_unverified_variant_count": 3 * len(questions) - len(evaluated),
        **calculate_recalls(evaluated),
        "details": details,
        "note": "Metrics include only variants explicitly marked human_verified.",
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
