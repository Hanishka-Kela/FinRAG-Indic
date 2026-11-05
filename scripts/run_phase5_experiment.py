import argparse
import asyncio
import json
from pathlib import Path

from finrag_indic.config import get_settings
from finrag_indic.evaluation.experimental import (
    Condition,
    ExperimentRunner,
    load_variants,
    load_probe_variants,
    summarize_results,
    write_reviewer_template,
)


async def run(args: argparse.Namespace) -> None:
    variants = load_variants(args.benchmark, limit=args.limit)
    if args.unsupported_benchmark is not None:
        variants.extend(load_probe_variants(args.unsupported_benchmark))
    conditions: tuple[Condition, ...] = (
        "llm_only", "rag_english", "rag_multilingual"
    )
    planned = len(variants) * len(conditions)
    print(f"Planned LLM requests: {planned}")
    if args.dry_run:
        print("Dry run: no provider or embedding calls made.")
        return
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    completed: set[tuple[str, str, str]] = set()
    if args.resume and output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                completed.add((item["canonical_id"], item["language"], item["condition"]))
    settings = get_settings()
    runner = ExperimentRunner(settings=settings)
    try:
        with output.open("a", encoding="utf-8") as handle:
            for variant in variants:
                for condition in conditions:
                    key = (variant.canonical_id, variant.language, condition)
                    if key in completed:
                        continue
                    record = await runner.run_case(variant, condition)
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
    finally:
        await runner.close()
    records = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = summarize_results(records)
    summary_path = output.with_name("phase5_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_reviewer_template(records, output.with_name("phase5_reviewer_scores.csv"))
    print(f"Completed records: {len(records)}")
    print(f"Summary: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 5 exploratory evaluation")
    parser.add_argument("--benchmark", type=Path, default=Path("configs/benchmarks/retrieval_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("results/raw/phase5_results.jsonl"))
    parser.add_argument("--limit", type=int, default=None, help="Limit canonical questions")
    parser.add_argument(
        "--unsupported-benchmark",
        type=Path,
        default=None,
        help="Optional separate exploratory abstention probes",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
