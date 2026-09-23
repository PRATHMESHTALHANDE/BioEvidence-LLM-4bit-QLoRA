"""Resilient JSON parsing and schema repair for biomedical model outputs."""

import json
import logging
import re
from typing import Any, Dict, Optional
from src.dataset.schema import DecisionType, StructuredModelOutput

logger = logging.getLogger(__name__)


def extract_json_block(text: str) -> str:
    """Extract raw JSON string from potential markdown code fences or conversational text."""
    clean = text.strip()
    # Check for markdown code fences ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Search for first { to last }
    start = clean.find("{")
    end = clean.rfind("}")
    if start != -1 and end != -1 and end > start:
        return clean[start : end + 1].strip()

    return clean


def repair_truncated_json(raw_text: str) -> Dict[str, Any]:
    """Attempt heuristic syntax repairs on truncated or slightly malformed JSON."""
    extracted = extract_json_block(raw_text)

    # 1. Direct parse attempt
    try:
        return json.loads(extracted)
    except json.JSONDecodeError:
        pass

    # 2. Repair common truncation issues (e.g. unclosed quotes or brackets)
    repaired = extracted
    if repaired.count('"') % 2 != 0:
        repaired += '"'
    if repaired.count("[") > repaired.count("]"):
        repaired += "]" * (repaired.count("[") - repaired.count("]"))
    if repaired.count("{") > repaired.count("}"):
        repaired += "}" * (repaired.count("{") - repaired.count("}"))

    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # 3. Fallback extraction using regex for key fields
    decision_match = re.search(r'"decision"\s*:\s*"(yes|no|maybe)"', extracted, re.I)
    answer_match = re.search(r'"answer"\s*:\s*"([^"]+)', extracted)

    decision = decision_match.group(1).lower() if decision_match else "maybe"
    answer = answer_match.group(1) if answer_match else "Extracted from partial model generation."

    return {
        "decision": decision,
        "answer": answer,
        "evidence": [],
        "uncertainty": "Output repaired from truncated response.",
        "limitations": [],
    }


def parse_and_validate_output(raw_response: str) -> StructuredModelOutput:
    """Parse raw LLM output into validated StructuredModelOutput schema."""
    parsed_dict = repair_truncated_json(raw_response)

    decision_raw = str(parsed_dict.get("decision", "maybe")).strip().lower()
    if decision_raw not in ("yes", "no", "maybe"):
        decision_raw = "maybe"

    return StructuredModelOutput(
        decision=DecisionType(decision_raw),
        answer=str(parsed_dict.get("answer", "No synthesis provided.")),
        evidence=list(parsed_dict.get("evidence", [])),
        uncertainty=parsed_dict.get("uncertainty"),
        limitations=list(parsed_dict.get("limitations", [])),
    )
