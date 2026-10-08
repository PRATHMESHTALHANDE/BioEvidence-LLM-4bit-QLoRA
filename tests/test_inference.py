"""Unit tests for inference parser and schema repair."""

from src.inference.schema_parser import (
    extract_json_block,
    parse_and_validate_output,
    repair_truncated_json,
)


def test_clean_json_extraction():
    raw = '```json\n{"decision": "yes", "answer": "Synthesized text", "evidence": ["quote 1"]}\n```'
    parsed = parse_and_validate_output(raw)
    assert parsed.decision.value == "yes"
    assert parsed.answer == "Synthesized text"
    assert len(parsed.evidence) == 1


def test_truncated_json_repair():
    # Intentionally unclosed quote and brace
    raw = '{"decision": "maybe", "answer": "Inconclusive results'
    repaired = repair_truncated_json(raw)
    assert repaired["decision"] == "maybe"
    assert "Inconclusive results" in repaired["answer"]


def test_markdown_code_fences_with_surrounding_chatter():
    raw = 'Here is the analysis:\n```json\n{"decision": "no", "answer": "No effect", "evidence": []}\n```\nHope that helps!'
    extracted = extract_json_block(raw)
    assert '{"decision": "no"' in extracted
