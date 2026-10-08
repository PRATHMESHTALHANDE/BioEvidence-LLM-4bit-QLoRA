"""Heuristic quality scoring and automated artifact filtering."""

from typing import List, Tuple

from src.dataset.schema import BiomedicalRecord

MIN_CONTEXT_CHARS = 50
MIN_ANSWER_CHARS = 10
BANNED_PHRASES = ["lorem ipsum", "todo", "test sentence", "placeholder"]


def filter_by_quality(records: List[BiomedicalRecord]) -> Tuple[List[BiomedicalRecord], int]:
    """Filter records by minimum length and sanity heuristics.

    Returns:
        passed_records: List of records passing all heuristic criteria.
        rejected_count: Number of rejected records.
    """
    passed: List[BiomedicalRecord] = []
    rejected = 0

    for r in records:
        ctx = r.context.strip()
        ans = r.answer.strip()

        if len(ctx) < MIN_CONTEXT_CHARS:
            rejected += 1
            continue

        if len(ans) < MIN_ANSWER_CHARS:
            rejected += 1
            continue

        # Check banned tokens
        lower_ctx = ctx.lower()
        if any(banned in lower_ctx for banned in BANNED_PHRASES):
            rejected += 1
            continue

        passed.append(r)

    return passed, rejected
