import json
from pathlib import Path

import pytest

from finrag_indic.config import Settings
from finrag_indic.evaluation.experimental import (
    ExperimentRunner,
    ExperimentVariant,
    citation_reference_validity,
    load_variants,
    summarize_results,
)
from finrag_indic.providers.mock import DeterministicMockProvider


def test_load_variants_preserves_exploratory_status() -> None:
    variants = load_variants(Path("configs/benchmarks/retrieval_v1.json"), limit=1)
    assert len(variants) == 3
    assert all(item.evidence_status == "candidate_unverified" for item in variants)


def test_citation_reference_validation_distinguishes_invalid_markers() -> None:
    evidence = [
        {
            "citation_id": "S1",
            "rank": 1,
            "score": 0.5,
            "chunk_id": "c",
            "document_id": "d",
            "title": "Title",
            "document_language": "en",
            "source_url": "https://example.com/a",
            "passage": "text",
        }
    ]
    from finrag_indic.contracts import EvidenceReference

    refs = [EvidenceReference.model_validate(evidence[0])]
    assert citation_reference_validity("Claim [S1].", refs) == (True, [])
    assert citation_reference_validity("Claim [S9].", refs) == (False, ["S9"])


@pytest.mark.anyio
async def test_llm_only_does_not_retrieve_or_attach_citations(tmp_path: Path) -> None:
    settings = Settings(
        environment="test",
        retrieval_index_path=tmp_path / "missing",
        _env_file=None,
    )
    runner = ExperimentRunner(settings=settings, provider=DeterministicMockProvider())
    variant = ExperimentVariant(
        canonical_id="q1",
        topic="interest",
        language="en",
        question="What is SI?",
        translation_status="canonical_english",
        evidence_status="candidate_unverified",
    )
    result = await runner.run_case(variant, "llm_only")
    assert result["http_success"]
    assert result["retrieval_index"] == "none"
    assert result["retrieved_chunks"] == []
    assert result["citation_count"] == 0


def test_summary_does_not_invent_missing_measurements() -> None:
    summary = summarize_results(
        [{"condition": "llm_only", "http_success": False, "total_ms": 3, "abstained": True, "answer_language_signal": False, "citation_reference_valid": False}]
    )
    assert summary[0]["mean_latency_successful_ms"] is None
    assert summary[0]["citation_valid_rate"] is None


def test_language_signal_requires_dominant_script_and_minimum_text() -> None:
    from finrag_indic.evaluation.experimental import _language_signal

    assert not _language_signal("OK", "en")
    assert not _language_signal("This is ठीक", "en")
    assert _language_signal("यह एक स्पष्ट उत्तर है", "hi")
