"""Unit tests for preprocessing, deduplication, and leakage prevention."""

from src.dataset.schema import BiomedicalRecord, Provenance
from src.preprocessing.deduplicate import deduplicate_records
from src.preprocessing.leakage import split_records_by_pmid
from src.preprocessing.normalize import clean_text


def test_clean_text():
    raw = "This is a  test   with ‘quotes’ and \u00a0non-breaking\n\n\n\nspaces."
    cleaned = clean_text(raw)
    assert "‘" not in cleaned
    assert "'" in cleaned
    assert "  " not in cleaned
    assert "\n\n\n" not in cleaned


def test_deduplicate_records():
    prov = Provenance(retrieval_date="2026-09-23")
    rec1 = BiomedicalRecord(
        id="1",
        source="test",
        source_id="1",
        context="Identical context",
        answer="Ans",
        provenance=prov,
    )
    rec2 = BiomedicalRecord(
        id="2",
        source="test",
        source_id="2",
        context="Identical context",
        answer="Ans",
        provenance=prov,
    )
    rec3 = BiomedicalRecord(
        id="3",
        source="test",
        source_id="3",
        context="Different context",
        answer="Ans",
        provenance=prov,
    )

    unique, count = deduplicate_records([rec1, rec2, rec3])
    assert len(unique) == 2
    assert count == 1


def test_leakage_prevention_zero_overlap():
    prov = Provenance(retrieval_date="2026-09-23")
    records = []
    # Create 20 records across 10 PMIDs (2 records per PMID)
    for i in range(10):
        pmid = f"PMID_{i}"
        records.append(
            BiomedicalRecord(
                id=f"rec_{i}_a",
                source="test",
                source_id=f"{i}_a",
                pmid=pmid,
                context=f"Context for {pmid} part A",
                answer="Answer A",
                provenance=prov,
            )
        )
        records.append(
            BiomedicalRecord(
                id=f"rec_{i}_b",
                source="test",
                source_id=f"{i}_b",
                pmid=pmid,
                context=f"Context for {pmid} part B",
                answer="Answer B",
                provenance=prov,
            )
        )

    train, ev, stats = split_records_by_pmid(records, train_ratio=0.7, random_seed=42)
    train_pmids = {r.pmid for r in train}
    eval_pmids = {r.pmid for r in ev}

    assert len(train_pmids.intersection(eval_pmids)) == 0
    assert stats["article_overlap_count"] == 0
