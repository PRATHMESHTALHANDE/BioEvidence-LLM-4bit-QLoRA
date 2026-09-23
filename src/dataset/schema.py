"""Canonical Pydantic Data Schemas for BioEvidence-LLM.

All ingested biomedical datasets (PubMedQA, MedQuAD, PubMed, PMC)
must strictly normalize into these schemas to ensure provenance,
reproducibility, and zero fact fabrication.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class DecisionType(str, Enum):
    """Normalized decision labels."""
    YES = "yes"
    NO = "no"
    MAYBE = "maybe"


class TaskTaxonomy(str, Enum):
    """Supported task taxonomy."""
    EVIDENCE_QA = "evidence_qa"
    EVIDENCE_CLASSIFICATION = "evidence_classification"
    PICO_EXTRACTION = "pico_extraction"
    EVIDENCE_SUMMARY = "evidence_summary"
    UNCERTAINTY_EXTRACTION = "uncertainty_extraction"
    LIMITATION_EXTRACTION = "limitation_extraction"
    MEDICAL_EXPLANATION = "medical_explanation"


class Provenance(BaseModel):
    """Tracks complete source provenance for auditability."""
    retrieval_date: str = Field(..., description="ISO retrieval date (YYYY-MM-DD)")
    source_url: Optional[str] = Field(None, description="Original URL where data was fetched")
    license_type: Optional[str] = Field(None, description="Source license, e.g. CC-BY-4.0")
    additional_metadata: Dict[str, Any] = Field(default_factory=dict)


class BiomedicalRecord(BaseModel):
    """Canonical ingested and normalized record contract."""
    id: str = Field(..., description="Unique record identifier, e.g. bioev_pubmedqa_12345")
    source: str = Field(..., description="Data source name (pubmedqa, medquad, pubmed, pmc)")
    source_id: str = Field(..., description="Original record ID within source")
    pmid: Optional[str] = Field(None, description="PubMed ID if available")
    pmcid: Optional[str] = Field(None, description="PubMed Central ID if available")
    task: TaskTaxonomy = Field(default=TaskTaxonomy.EVIDENCE_QA)
    domain: Optional[str] = Field(None, description="Biomedical domain (e.g. oncology, pharmacology)")
    context: str = Field(..., description="Source biomedical evidence text / abstract excerpt")
    question: Optional[str] = Field(None, description="Biomedical research query")
    answer: str = Field(..., description="Evidence-grounded structured response")
    decision: Optional[DecisionType] = Field(None, description="Decision: yes, no, or maybe")
    evidence: List[str] = Field(default_factory=list, description="Verbatim cited evidence excerpts")
    uncertainty: Optional[str] = Field(None, description="Preserved study uncertainty / caveats")
    limitations: List[str] = Field(default_factory=list, description="Methodological or sample limitations")
    language: str = Field(default="en")
    license: Optional[str] = Field(default="CC-BY-4.0")
    provenance: Provenance = Field(..., description="Full provenance metadata")
    quality_score: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("context")
    @classmethod
    def validate_non_empty_context(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Evidence context must not be empty or solely whitespace.")
        return clean

    @field_validator("answer")
    @classmethod
    def validate_non_empty_answer(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Answer must not be empty or solely whitespace.")
        return clean


class PICORecord(BaseModel):
    """PICO structured extraction contract."""
    population: str = Field(..., description="Patient or population studied")
    intervention: str = Field(..., description="Intervention, treatment, or exposure")
    comparison: Optional[str] = Field(None, description="Comparison group or placebo")
    outcome: str = Field(..., description="Measured clinical or biological outcome")


class StructuredModelOutput(BaseModel):
    """Contract for fine-tuned LLM inference response."""
    decision: DecisionType
    answer: str
    evidence: List[str] = Field(default_factory=list)
    uncertainty: Optional[str] = None
    limitations: List[str] = Field(default_factory=list)
