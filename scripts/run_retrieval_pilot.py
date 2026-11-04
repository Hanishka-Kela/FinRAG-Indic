import json
from argparse import ArgumentParser
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from finrag_indic.config import get_settings
from finrag_indic.retrieval.embeddings import BgeM3Embedder


class PilotQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    language: str
    question: str
    status: str = "exploratory_unverified"
    gold_chunk_ids: list[str] = Field(default_factory=list)


def main() -> None:
    parser = ArgumentParser(description="Run exploratory multilingual retrieval queries")
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("configs/pilots/retrieval_questions.json"),
    )
    parser.add_argument("--index-root", type=Path, default=Path("data/indexes"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/raw/retrieval_pilot.jsonl")
    )
    args = parser.parse_args()
    settings = get_settings()
    questions = [
        PilotQuestion.model_validate(item)
        for item in json.loads(args.questions.read_text(encoding="utf-8"))
    ]
    embedder = BgeM3Embedder(
        model_id=settings.embedding_model,
        revision=settings.embedding_revision,
        cache_folder=Path("data/models/sentence-transformers"),
        device=settings.embedding_device,
    )
    query_vectors = embedder.embed(
        [question.question for question in questions],
        batch_size=settings.embedding_batch_size,
    )
    from finrag_indic.retrieval.index import FaissRetriever

    indexes = {
        "english": FaissRetriever.load(
            args.index_root / "english", expected_embedding=embedder.metadata
        ),
        "multilingual": FaissRetriever.load(
            args.index_root / "multilingual", expected_embedding=embedder.metadata
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for question, query in zip(questions, query_vectors):
            targets = ["english"] if question.language == "en" else ["multilingual", "english"]
            for target in targets:
                hits = indexes[target].search(query, top_k=settings.retrieval_top_k)
                record = {
                    "recorded_at": datetime.now(UTC).isoformat(),
                    "question": question.model_dump(),
                    "corpus_view": target,
                    "retrieval_mode": (
                        "direct_multilingual_to_english"
                        if question.language != "en" and target == "english"
                        else "same_or_multilingual_corpus"
                    ),
                    "embedding": embedder.metadata.model_dump(),
                    "hits": [hit.model_dump(mode="json") for hit in hits],
                    "recall_at_k": None,
                    "recall_note": "Not calculated: no manually verified gold chunk mapping.",
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(
                    f"{question.question_id} -> {target}: "
                    f"{hits[0].chunk.chunk_id if hits else 'no hit'}"
                )


if __name__ == "__main__":
    main()

