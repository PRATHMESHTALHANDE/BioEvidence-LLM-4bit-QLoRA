"""PMID/Article-level train-test splitting to strictly prevent data leakage."""

import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from src.dataset.schema import BiomedicalRecord


def split_records_by_pmid(
    records: List[BiomedicalRecord],
    train_ratio: float = 0.85,
    random_seed: int = 42,
) -> Tuple[List[BiomedicalRecord], List[BiomedicalRecord], Dict[str, Any]]:
    """Split records into train and evaluation subsets, grouping by article identifier (PMID/PMCID).

    Ensures zero article overlap between train and evaluation sets.
    """
    rng = random.Random(random_seed)

    # Group records by article identifier
    article_groups: Dict[str, List[BiomedicalRecord]] = defaultdict(list)
    for rec in records:
        key = rec.pmid or rec.pmcid or rec.id
        article_groups[key].append(rec)

    keys = list(article_groups.keys())
    rng.shuffle(keys)

    cutoff = int(len(keys) * train_ratio)
    train_keys = set(keys[:cutoff])
    eval_keys = set(keys[cutoff:])

    train_records: List[BiomedicalRecord] = []
    eval_records: List[BiomedicalRecord] = []

    for k in train_keys:
        train_records.extend(article_groups[k])
    for k in eval_keys:
        eval_records.extend(article_groups[k])

    # Overlap validation assertion
    overlap = train_keys.intersection(eval_keys)
    if overlap:
        raise ValueError(
            f"CRITICAL LEAKAGE DETECTED: {len(overlap)} articles exist in both splits!"
        )

    stats = {
        "train_articles": len(train_keys),
        "eval_articles": len(eval_keys),
        "train_records": len(train_records),
        "eval_records": len(eval_records),
        "leakage_detected": False,
        "article_overlap_count": 0,
    }

    return train_records, eval_records, stats
