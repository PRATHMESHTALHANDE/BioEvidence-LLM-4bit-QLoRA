"""Exact SHA-256 and near-duplicate filtering."""

import hashlib
from typing import List, Set, Tuple
from src.dataset.schema import BiomedicalRecord


def compute_sha256(text: str) -> str:
    """Calculate SHA-256 hash of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def deduplicate_records(
    records: List[BiomedicalRecord],
) -> Tuple[List[BiomedicalRecord], int]:
    """Remove exact duplicate records based on SHA-256 of context + question.

    Returns:
        unique_records: List of deduplicated BiomedicalRecord.
        num_duplicates_removed: Count of removed duplicate items.
    """
    seen_hashes: Set[str] = set()
    unique: List[BiomedicalRecord] = []
    duplicates_count = 0

    for rec in records:
        content_key = f"{rec.question or ''}:::{rec.context.strip()}"
        h = compute_sha256(content_key)
        if h in seen_hashes:
            duplicates_count += 1
            continue
        seen_hashes.add(h)
        unique.append(rec)

    return unique, duplicates_count
