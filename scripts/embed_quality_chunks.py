import json
import hashlib
import os
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from finrag_indic.config import get_settings
from finrag_indic.knowledge.models import Chunk
from finrag_indic.retrieval.embeddings import BgeM3Embedder


def main() -> None:
    parser = ArgumentParser(description="Embed prepared Phase 3.5 chunks in a clean process")
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("data/processed/multilingual-genuine-v2.chunks.jsonl"),
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
        "--report", type=Path, default=Path("data/processed/phase3_5_extraction_report.json")
    )
    parser.add_argument("--part-size", type=int, default=40)
    parser.add_argument("--part-start", type=int, default=None, help="Internal worker option")
    parser.add_argument("--part-stop", type=int, default=None, help="Internal worker option")
    parser.add_argument("--part-output", type=Path, default=None, help="Internal worker option")
    args = parser.parse_args()
    chunks = [
        Chunk.model_validate_json(line)
        for line in args.chunks.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    settings = get_settings()
    embedder = BgeM3Embedder(
        model_id=settings.embedding_model,
        revision=settings.embedding_revision,
        cache_folder=Path("data/models/sentence-transformers"),
        device=settings.embedding_device,
    )
    if args.part_start is not None or args.part_stop is not None or args.part_output is not None:
        if args.part_start is None or args.part_stop is None or args.part_output is None:
            raise SystemExit("All internal part options must be supplied together")
        vectors = np.vstack(
            [
                embedder.embed([chunk.text], batch_size=1)[0]
                for chunk in chunks[args.part_start : args.part_stop]
            ]
        ).astype(np.float32)
        args.part_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.part_output, vectors, allow_pickle=False)
        print(f"Embedded rows {args.part_start}:{args.part_stop}")
        return

    texts = [chunk.text for chunk in chunks]
    digest = hashlib.sha256(
        json.dumps(
            {
                "model_id": embedder.metadata.model_id,
                "revision": embedder.metadata.revision,
                "texts": texts,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    vectors = None
    if args.embedding_cache.exists():
        cached = np.load(args.embedding_cache, allow_pickle=False)
        if str(cached["digest"].item()) == digest:
            candidate = np.asarray(cached["vectors"], dtype=np.float32)
            if candidate.shape == (len(chunks), embedder.metadata.dimension):
                vectors = candidate
    if vectors is None:
        parts_directory = args.embedding_cache.with_suffix(".parts")
        part_paths: list[Path] = []
        for start in range(0, len(chunks), args.part_size):
            stop = min(start + args.part_size, len(chunks))
            part_path = parts_directory / f"part-{start:04d}-{stop:04d}.npy"
            part_paths.append(part_path)
            if part_path.exists():
                part = np.load(part_path, allow_pickle=False)
                if part.shape == (stop - start, embedder.metadata.dimension):
                    continue
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--chunks",
                str(args.chunks),
                "--part-start",
                str(start),
                "--part-stop",
                str(stop),
                "--part-output",
                str(part_path),
            ]
            subprocess.run(command, check=True, env=os.environ.copy())
        vectors = np.vstack(
            [np.load(part_path, allow_pickle=False) for part_path in part_paths]
        ).astype(np.float32)
        args.embedding_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.embedding_cache, digest=np.array(digest), vectors=vectors
        )
    from finrag_indic.retrieval.index import FaissRetriever

    retriever = FaissRetriever.build(
        chunks=chunks,
        vectors=vectors,
        embedding=embedder.metadata,
        corpus_view="multilingual-genuine-v2",
    )
    retriever.save(args.output)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    report["embedding"] = embedder.metadata.model_dump(mode="json")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Built {args.output} with {len(chunks)} vectors")


if __name__ == "__main__":
    main()
