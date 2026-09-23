"""System prompts, instruction templates, and JSON target formatters for BioEvidence-LLM."""

import json
from typing import Any, Dict, Optional
from src.dataset.schema import BiomedicalRecord, StructuredModelOutput


BIOEVIDENCE_SYSTEM_PROMPT = """You are BioEvidence-LLM, an expert biomedical literature analysis assistant.
Your role is to analyze biomedical questions and provided source evidence strictly and accurately.

CRITICAL OPERATIONAL RULES:
1. EVIDENCE BEFORE GENERATION: Base all claims solely on the provided evidence text. Never invent facts, biological mechanisms, sample sizes, or p-values.
2. PRESERVE UNCERTAINTY: If the source study is preliminary, inconclusive, or has mixed findings, classify as "maybe" and document the nuance under "uncertainty".
3. PRESERVE EXACT NUMERICAL VALUES: Retain exact statistical numbers, odds ratios, dosages, and confidence intervals verbatim.
4. STRUCTURED OUTPUT: Always reply strictly with a valid JSON object matching this schema:
{
  "decision": "yes" | "no" | "maybe",
  "answer": "Concise, evidence-grounded synthesis answering the question.",
  "evidence": ["Verbatim sentence or phrase from source text supporting the answer."],
  "uncertainty": "Explicit study caveats, ambiguous results, or limited generalizability (or null if none).",
  "limitations": ["Author-reported study limitations, e.g. sample size, retrospective design."]
}
Do not include any conversational filler, markdown backticks around the JSON, or text outside the JSON object.
"""


def build_user_prompt(question: Optional[str], context: str, task: str = "evidence_qa") -> str:
    """Format the user instruction prompt.

    Args:
        question: The biomedical query.
        context: The source biomedical evidence (abstract/excerpt).
        task: The specific NLP task.

    Returns:
        Formatted instruction string.
    """
    clean_context = context.strip()
    if question and question.strip():
        clean_question = question.strip()
        return (
            f"Task: {task}\n\n"
            f"Evidence Context:\n{clean_context}\n\n"
            f"Biomedical Question:\n{clean_question}\n\n"
            f"Analyze the evidence and answer the question in the required JSON format."
        )
    return (
        f"Task: {task}\n\n"
        f"Evidence Context:\n{clean_context}\n\n"
        f"Extract the evidence findings, uncertainty, and limitations in the required JSON format."
    )


def format_target_json(record: BiomedicalRecord) -> str:
    """Convert a BiomedicalRecord into the canonical JSON target string for SFT training."""
    output = StructuredModelOutput(
        decision=record.decision if record.decision else "maybe",
        answer=record.answer,
        evidence=record.evidence,
        uncertainty=record.uncertainty,
        limitations=record.limitations,
    )
    return json.dumps(output.model_dump(), ensure_ascii=False, indent=2)


def format_chat_sample(record: BiomedicalRecord) -> Dict[str, Any]:
    """Format a record into standard Hugging Face messages format (ChatML-compatible)."""
    user_content = build_user_prompt(
        question=record.question,
        context=record.context,
        task=record.task.value if hasattr(record.task, "value") else str(record.task),
    )
    assistant_content = format_target_json(record)

    return {
        "messages": [
            {"role": "system", "content": BIOEVIDENCE_SYSTEM_PROMPT.strip()},
            {"role": "user", "content": user_content.strip()},
            {"role": "assistant", "content": assistant_content.strip()},
        ]
    }
