from pydantic import BaseModel, ConfigDict, Field, model_validator


class QuestionVariant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    translation_status: str
    evidence_status: str = "candidate_unverified"
    verified_gold_chunk_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def verified_status_requires_evidence(self) -> "QuestionVariant":
        if self.evidence_status == "human_verified" and not self.verified_gold_chunk_ids:
            raise ValueError("human_verified evidence requires at least one gold chunk ID")
        if self.evidence_status != "human_verified" and self.verified_gold_chunk_ids:
            raise ValueError("gold chunk IDs are allowed only after human verification")
        return self


class BenchmarkQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    canonical_id: str
    topic: str
    variants: dict[str, QuestionVariant]

    @model_validator(mode="after")
    def requires_three_languages(self) -> "BenchmarkQuestion":
        if set(self.variants) != {"en", "hi", "ta"}:
            raise ValueError("each benchmark question requires en, hi, and ta variants")
        return self


def recall_at_k(
    evaluated: list[tuple[set[str], list[str]]], k: int
) -> float | None:
    if not evaluated:
        return None
    hits = sum(bool(gold.intersection(retrieved[:k])) for gold, retrieved in evaluated)
    return hits / len(evaluated)


def calculate_recalls(
    evaluated: list[tuple[set[str], list[str]]]
) -> dict[str, float | None]:
    return {f"recall_at_{k}": recall_at_k(evaluated, k) for k in (1, 3, 5)}
