"""Unit tests for Canonical BiomedicalRecord schema and contracts."""

import pytest
from pydantic import ValidationError
from src.dataset.schema import (
    BiomedicalRecord,
    DecisionType,
    Provenance,
    StructuredModelOutput,
    TaskTaxonomy,
)


def test_valid_biomedical_record():
    """Test creating a fully populated valid BiomedicalRecord."""
    record = BiomedicalRecord(
        id="bioev_pubmedqa_12345",
        source="pubmedqa",
        source_id="12345678",
        pmid="12345678",
        pmcid=None,
        task=TaskTaxonomy.EVIDENCE_QA,
        domain="pharmacology",
        context="Background: Study on drug X. Results: Significant reduction in mortality (p=0.01). Conclusion: Effective.",
        question="Does drug X reduce mortality?",
        answer="The evidence indicates that drug X significantly reduces mortality.",
        decision=DecisionType.YES,
        evidence=["Significant reduction in mortality (p=0.01)."],
        uncertainty="The study was single-center.",
        limitations=["Small sample size (n=120)"],
        language="en",
        license="CC-BY-4.0",
        provenance=Provenance(
            retrieval_date="2026-09-23",
            source_url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
        ),
        quality_score=0.98,
    )
    assert record.id == "bioev_pubmedqa_12345"
    assert record.decision == DecisionType.YES
    assert len(record.evidence) == 1
    assert record.provenance.retrieval_date == "2026-09-23"


def test_empty_context_validation():
    """Test that empty or whitespace-only context raises ValidationError."""
    with pytest.raises(ValidationError):
        BiomedicalRecord(
            id="bioev_test_1",
            source="test",
            source_id="1",
            context="   ",  # Invalid empty context
            answer="Valid answer",
            provenance=Provenance(retrieval_date="2026-09-23"),
        )


def test_empty_answer_validation():
    """Test that empty or whitespace-only answer raises ValidationError."""
    with pytest.raises(ValidationError):
        BiomedicalRecord(
            id="bioev_test_2",
            source="test",
            source_id="2",
            context="Valid context text from scientific abstract.",
            answer="",  # Invalid empty answer
            provenance=Provenance(retrieval_date="2026-09-23"),
        )


def test_structured_model_output_schema():
    """Test StructuredModelOutput contract."""
    output = StructuredModelOutput(
        decision=DecisionType.MAYBE,
        answer="The study results were inconclusive.",
        evidence=["No statistically significant difference was observed (p=0.12)."],
        uncertainty="High confidence intervals.",
        limitations=["Retrospective study"],
    )
    dumped = output.model_dump()
    assert dumped["decision"] == "maybe"
    assert len(dumped["evidence"]) == 1
    assert dumped["uncertainty"] == "High confidence intervals."
