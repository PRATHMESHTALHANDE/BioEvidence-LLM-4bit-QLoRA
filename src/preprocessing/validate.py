"""Schema conformance and contract validation."""

from typing import Any, Dict, List, Tuple
from pydantic import ValidationError
from src.dataset.schema import BiomedicalRecord


def validate_records(records_raw: List[Dict[str, Any]]) -> Tuple[List[BiomedicalRecord], List[Dict[str, Any]]]:
    """Validate a list of raw dicts against BiomedicalRecord schema.

    Returns:
        valid_records: List of validated BiomedicalRecord instances.
        failed_records: List of dicts containing the original item and error details.
    """
    valid: List[BiomedicalRecord] = []
    failed: List[Dict[str, Any]] = []

    for item in records_raw:
        try:
            rec = BiomedicalRecord.model_validate(item)
            valid.append(rec)
        except ValidationError as e:
            failed.append({
                "item_id": item.get("id", "unknown"),
                "errors": e.errors(),
            })

    return valid, failed
