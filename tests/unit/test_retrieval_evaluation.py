import pytest
import json
from pathlib import Path
from pydantic import ValidationError

from finrag_indic.evaluation.retrieval import (
    QuestionVariant,
    calculate_recalls,
)


def test_recalls_are_calculated_at_required_cutoffs() -> None:
    evaluated = [
        ({"gold-a"}, ["gold-a", "x", "y", "z", "q"]),
        ({"gold-b"}, ["x", "y", "gold-b", "z", "q"]),
        ({"gold-c"}, ["x", "y", "z", "q", "gold-c"]),
        ({"missing"}, ["x", "y", "z", "q", "r"]),
    ]

    assert calculate_recalls(evaluated) == {
        "recall_at_1": 0.25,
        "recall_at_3": 0.5,
        "recall_at_5": 0.75,
    }


def test_no_verified_questions_produces_no_metrics() -> None:
    assert calculate_recalls([]) == {
        "recall_at_1": None,
        "recall_at_3": None,
        "recall_at_5": None,
    }


def test_unverified_variant_cannot_contain_gold_evidence() -> None:
    with pytest.raises(ValidationError, match="only after human verification"):
        QuestionVariant(
            question="test",
            translation_status="draft_unverified",
            evidence_status="candidate_unverified",
            verified_gold_chunk_ids=["chunk-1"],
        )


def test_benchmark_is_review_gated_and_trilingual() -> None:
    payload = json.loads(
        Path("configs/benchmarks/retrieval_v1.json").read_text(encoding="utf-8")
    )

    assert len(payload) == 15
    for item in payload:
        assert set(item["variants"]) == {"en", "hi", "ta"}
        for variant in item["variants"].values():
            assert variant["evidence_status"] == "candidate_unverified"
            assert variant["verified_gold_chunk_ids"] == []
