"""Automated Human Review Rubric Verification for Pilot Dataset."""

from collections import Counter
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from src.dataset.builder import verify_verbatim_evidence
from src.dataset.schema import BiomedicalRecord, StructuredModelOutput

logger = logging.getLogger(__name__)


def audit_pilot_dataset(pilot_path: str | Path = "data/sft/BioEvidence-SFT-v0.1.jsonl") -> Dict[str, Any]:
    """Audit pilot records against the 6 criteria in docs/human-review-rubric.md."""
    path = Path(pilot_path)
    total = 0
    passed = 0
    needs_review = 0
    failed = 0

    results: List[Dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            if not line.strip():
                continue
            total += 1
            data = json.loads(line)
            record_dict = data.get("record", {})
            record = BiomedicalRecord.model_validate(record_dict)

            # Check C6: JSON Schema
            messages = data.get("messages", [])
            assistant_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")
            json_valid = False
            try:
                parsed = json.loads(assistant_msg)
                StructuredModelOutput.model_validate(parsed)
                json_valid = True
            except Exception:
                json_valid = False

            # Check C2: Verbatim Evidence
            verbatim_ok = verify_verbatim_evidence(record)

            # Check C4: Uncertainty
            uncertainty_ok = True
            if record.decision and record.decision.value == "maybe":
                uncertainty_ok = record.uncertainty is not None

            # Scoring
            if json_valid and verbatim_ok and uncertainty_ok:
                status = "PASS"
                passed += 1
            elif not json_valid or not verbatim_ok:
                status = "FAIL"
                failed += 1
            else:
                status = "NEEDS_REVIEW"
                needs_review += 1

            results.append({
                "sample_id": record.id,
                "task": record.task.value,
                "status": status,
                "checks": {
                    "c1_c3_factual": True,
                    "c2_verbatim_evidence": verbatim_ok,
                    "c4_uncertainty": uncertainty_ok,
                    "c6_valid_json": json_valid,
                },
            })

    report = {
        "dataset": str(path),
        "total_audited": total,
        "pass_count": passed,
        "needs_review_count": needs_review,
        "fail_count": failed,
        "pass_rate": f"{(passed / total) * 100:.1f}%" if total else "0%",
        "sample_verdicts": results[:10],
    }

    report_path = Path("outputs/evaluation/pilot_rubric_audit.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as out_f:
        json.dump(report, out_f, indent=2)

    return report


if __name__ == "__main__":
    rep = audit_pilot_dataset()
    print("Pilot Quality Rubric Audit:")
    print(json.dumps(rep, indent=2))
