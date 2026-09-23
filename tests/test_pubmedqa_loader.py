"""Unit tests for PubMedQA loader and normalizer."""

import json
from pathlib import Path
from src.data.pubmedqa_loader import PubMedQALoader
from src.dataset.schema import DecisionType, TaskTaxonomy


def test_pubmedqa_normalization_sample(tmp_path: Path):
    """Test normalizing a mocked PubMedQA record."""
    loader = PubMedQALoader(raw_dir=tmp_path / "raw", interim_dir=tmp_path / "interim")
    mock_raw_item = {
        "QUESTION": "Does treatment with statins reduce cardiovascular events in diabetic patients?",
        "CONTEXTS": [
            "Cardiovascular disease is the leading cause of morbidity.",
            "We conducted a randomized trial with 500 patients.",
            "Statin therapy resulted in a 35% reduction in cardiovascular events (p < 0.001).",
            "Statins significantly reduce cardiovascular events in diabetic populations.",
        ],
        "LABELS": ["BACKGROUND", "METHODS", "RESULTS", "CONCLUSIONS"],
        "MESHES": ["Hydroxymethylglutaryl-CoA Reductase Inhibitors", "Diabetes Mellitus"],
        "YEAR": "2020",
        "final_decision": "yes",
        "LONG_ANSWER": "Statins demonstrate significant clinical efficacy in reducing cardiovascular events.",
    }

    record = loader.normalize_record("12345678", mock_raw_item, "2026-09-23")
    assert record.id == "bioev_pubmedqa_12345678"
    assert record.pmid == "12345678"
    assert record.decision == DecisionType.YES
    assert record.task == TaskTaxonomy.EVIDENCE_QA
    assert "BACKGROUND:" in record.context
    assert len(record.evidence) >= 1
    assert "Statins significantly reduce" in record.evidence[0]


def test_pubmedqa_maybe_uncertainty(tmp_path: Path):
    """Test that 'maybe' decisions preserve uncertainty."""
    loader = PubMedQALoader(raw_dir=tmp_path / "raw", interim_dir=tmp_path / "interim")
    mock_raw_item = {
        "QUESTION": "Is drug Y superior to drug Z?",
        "CONTEXTS": ["Trial showed no statistically significant difference."],
        "LABELS": ["RESULTS"],
        "final_decision": "maybe",
        "LONG_ANSWER": "Results remain inconclusive.",
    }

    record = loader.normalize_record("87654321", mock_raw_item, "2026-09-23")
    assert record.decision == DecisionType.MAYBE
    assert record.uncertainty is not None
    assert "inconclusive" in record.uncertainty.lower()
