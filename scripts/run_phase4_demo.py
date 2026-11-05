import json
from argparse import ArgumentParser
from datetime import UTC, datetime
from pathlib import Path

import httpx

from finrag_indic.api.app import create_app
from finrag_indic.config import get_settings


DEMO_IDS = {"fin-001", "fin-002", "fin-007", "fin-008", "fin-014"}


async def run_demo(*, benchmark: Path, output: Path, use_configured_provider: bool) -> None:
    questions = json.loads(benchmark.read_text(encoding="utf-8"))
    requests = [
        {
            "canonical_id": item["canonical_id"],
            "topic": item["topic"],
            "question": variant["question"],
            "language": language,
        }
        for item in questions
        if item["canonical_id"] in DEMO_IDS
        for language, variant in item["variants"].items()
    ]
    settings = get_settings()
    if not use_configured_provider:
        settings = settings.model_copy(update={"llm_provider": "mock"})
    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)
    records = []
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://phase4-demo", timeout=180
        ) as client:
            for item in requests:
                response = await client.post(
                    "/chat",
                    json={"question": item["question"], "language": item["language"]},
                )
                records.append(
                    {
                        **item,
                        "http_status": response.status_code,
                        "response": response.json(),
                    }
                )
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "provider_mode": "configured" if use_configured_provider else "mock",
        "note": (
            "Mock answers are deterministic engineering outputs, not hosted-LLM quality examples. "
            "Retrieved passages and citations are from the real v2 index."
        ),
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} Phase 4 demonstration records to {output}")


def main() -> None:
    import asyncio

    parser = ArgumentParser(description="Run the Phase 4 multilingual chatbot demo")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("configs/benchmarks/retrieval_v1.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/raw/phase4_demo.json")
    )
    parser.add_argument(
        "--use-configured-provider",
        action="store_true",
        help="Use FINRAG_LLM_PROVIDER instead of the deterministic mock",
    )
    args = parser.parse_args()
    asyncio.run(
        run_demo(
            benchmark=args.benchmark,
            output=args.output,
            use_configured_provider=args.use_configured_provider,
        )
    )


if __name__ == "__main__":
    main()
