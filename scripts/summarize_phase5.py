import csv
import json
from argparse import ArgumentParser
from pathlib import Path

from PIL import Image, ImageDraw

from finrag_indic.evaluation.experimental import summarize_results, write_reviewer_template


def _bar_chart(rows: list[dict[str, object]], path: Path) -> None:
    width, height = 1000, 620
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title = "Phase 5 response success and latency"
    draw.text((40, 30), title, fill="black")
    max_latency = max(
        [float(row["mean_latency_ms"]) for row in rows if row["mean_latency_ms"] is not None]
        or [1]
    )
    bar_width = 220
    for index, row in enumerate(rows):
        x = 70 + index * 300
        success = float(row["success_rate"] or 0)
        latency = float(row["mean_latency_ms"] or 0)
        draw.rectangle((x, 500 - int(success * 300), x + bar_width, 500), fill="#4472C4")
        draw.text((x, 515), str(row["condition"]), fill="black")
        draw.text((x, 475 - int(success * 300)), f"success {success:.0%}", fill="black")
        latency_height = int((latency / max_latency) * 220)
        draw.rectangle(
            (x + 25, 450 - latency_height, x + 75, 450), fill="#ED7D31"
        )
        draw.text((x + 5, 430 - latency_height), f"{latency:.0f} ms", fill="black")
    draw.text((40, 580), "Blue: success rate (0-100%); orange: mean latency (relative scale)", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def main() -> None:
    parser = ArgumentParser(description="Summarize collected Phase 5 records")
    parser.add_argument("--input", type=Path, default=Path("results/raw/phase5_results.jsonl"))
    args = parser.parse_args()
    input_path = args.input
    records = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = summarize_results(records)
    output_dir = Path("results/aggregate")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase5_summary.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "phase5_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else ["condition"])
        writer.writeheader()
        writer.writerows(rows)
    write_reviewer_template(records, output_dir / "phase5_reviewer_scores.csv")
    _bar_chart(rows, Path("results/figures/phase5_response_performance.png"))


if __name__ == "__main__":
    main()
