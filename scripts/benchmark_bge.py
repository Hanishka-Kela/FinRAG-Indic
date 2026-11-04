import json
from argparse import ArgumentParser
from datetime import UTC, datetime
from pathlib import Path

from finrag_indic.config import get_settings
from finrag_indic.retrieval.embeddings import BgeM3Embedder, benchmark_embeddings


PILOT_TEXTS = [
    "UPI payment safety",
    "यूपीआई भुगतान सुरक्षा",
    "UPI கட்டண பாதுகாப்பு",
]


def main() -> None:
    parser = ArgumentParser(description="Benchmark pinned BGE-M3 on three short texts")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/raw/bge_m3_local_benchmark.json"),
    )
    args = parser.parse_args()
    settings = get_settings()
    provider = BgeM3Embedder(
        model_id=settings.embedding_model,
        revision=settings.embedding_revision,
        cache_folder=Path("data/models/sentence-transformers"),
        device=settings.embedding_device,
    )
    vectors, run = benchmark_embeddings(provider, PILOT_TEXTS, batch_size=3)
    payload = {
        **run.model_dump(mode="json"),
        "recorded_at": datetime.now(UTC).isoformat(),
        "vector_shape": list(vectors.shape),
        "device": settings.embedding_device,
        "measurement_note": (
            "rss_delta_bytes is process RSS after the operation minus RSS before it; "
            "it is not a hardware-wide or sampled peak-memory measurement."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
