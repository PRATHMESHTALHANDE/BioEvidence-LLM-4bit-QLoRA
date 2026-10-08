"""PubMedQA Data Loader and Normalizer for BioEvidence-LLM.

Fetches the official PubMedQA labeled dataset (ori_pqal.json), saves
the immutable raw copy into data/raw/pubmedqa/, normalizes each sample
into the canonical BiomedicalRecord schema, and generates an ingestion report.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from src.dataset.schema import BiomedicalRecord, DecisionType, Provenance, TaskTaxonomy

logger = logging.getLogger(__name__)

PUBMEDQA_ORIGINAL_URL = (
    "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"
)
PUBMEDQA_TEST_SPLIT_URL = (
    "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/test_ground_truth.json"
)


class PubMedQALoader:
    """Ingestion and normalization handler for PubMedQA."""

    def __init__(
        self,
        raw_dir: str | Path = "data/raw/pubmedqa",
        interim_dir: str | Path = "data/interim/pubmedqa",
    ):
        self.raw_dir = Path(raw_dir)
        self.interim_dir = Path(interim_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.interim_dir.mkdir(parents=True, exist_ok=True)

    def download_raw(
        self, url: str = PUBMEDQA_ORIGINAL_URL, filename: str = "ori_pqal.json"
    ) -> Path:
        """Download raw dataset immutably if not already present."""
        target_path = self.raw_dir / filename
        if target_path.exists():
            logger.info("Using cached raw PubMedQA data at %s", target_path)
            return target_path

        logger.info("Downloading PubMedQA from %s to %s", url, target_path)
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with open(target_path, "w", encoding="utf-8") as f:
            f.write(response.text)

        logger.info("Downloaded %d bytes to %s", target_path.stat().st_size, target_path)
        return target_path

    def load_raw(self, filepath: Optional[Path] = None) -> Dict[str, Any]:
        """Load the raw JSON dictionary from disk."""
        target = filepath or (self.raw_dir / "ori_pqal.json")
        if not target.exists():
            target = self.download_raw()

        with open(target, "r", encoding="utf-8") as f:
            return json.load(f)

    def normalize_record(
        self, pmid: str, raw_item: Dict[str, Any], retrieval_date: str
    ) -> BiomedicalRecord:
        """Normalize a single raw PubMedQA record into canonical BiomedicalRecord."""
        question = raw_item.get("QUESTION", "").strip()
        contexts = raw_item.get("CONTEXTS", [])
        context_labels = raw_item.get("LABELS", [])

        # Build clean labeled context if labels exist
        context_parts = []
        if context_labels and len(context_labels) == len(contexts):
            for label, text in zip(context_labels, contexts):
                context_parts.append(f"{label}: {text.strip()}")
        else:
            context_parts = [c.strip() for c in contexts]

        full_context = "\n\n".join(context_parts)
        if not full_context:
            full_context = "No abstract text provided."

        long_answer = raw_item.get("LONG_ANSWER", "").strip()
        decision_raw = raw_item.get("final_decision", "").strip().lower()

        # Map decision string to DecisionType enum
        decision: Optional[DecisionType] = None
        if decision_raw in ("yes", "no", "maybe"):
            decision = DecisionType(decision_raw)

        # Extract direct evidence sentence from RESULTS or CONCLUSION if available
        evidence: List[str] = []
        if contexts:
            # Prefer conclusions or results context for direct quotation
            target_evidence = contexts[-1].strip()
            if target_evidence:
                evidence.append(target_evidence)

        # Identify uncertainty markers in maybe decisions
        uncertainty = None
        if decision == DecisionType.MAYBE:
            uncertainty = (
                "The study findings are preliminary, inconclusive, or report conflicting outcomes."
            )

        provenance = Provenance(
            retrieval_date=retrieval_date,
            source_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            license_type="MIT",
            additional_metadata={
                "meshes": raw_item.get("MESHES", []),
                "year": raw_item.get("YEAR"),
            },
        )

        return BiomedicalRecord(
            id=f"bioev_pubmedqa_{pmid}",
            source="pubmedqa",
            source_id=pmid,
            pmid=pmid,
            pmcid=None,
            task=TaskTaxonomy.EVIDENCE_QA,
            domain="clinical_medicine",
            context=full_context,
            question=question,
            answer=long_answer if long_answer else f"Study concluded with {decision_raw}.",
            decision=decision,
            evidence=evidence,
            uncertainty=uncertainty,
            limitations=[],
            language="en",
            license="MIT",
            provenance=provenance,
            quality_score=1.0,
        )

    def process_and_save(self) -> Dict[str, Any]:
        """Process full PubMedQA dataset, save normalized records, and generate ingestion report."""
        raw_data = self.load_raw()
        retrieval_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        normalized_records: List[BiomedicalRecord] = []
        decision_counts = {"yes": 0, "no": 0, "maybe": 0, "unknown": 0}

        for pmid, item in raw_data.items():
            record = self.normalize_record(pmid, item, retrieval_date)
            normalized_records.append(record)
            if record.decision:
                decision_counts[record.decision.value] += 1
            else:
                decision_counts["unknown"] += 1

        # Save interim normalized JSONL
        output_jsonl = self.interim_dir / "pubmedqa_normalized.jsonl"
        with open(output_jsonl, "w", encoding="utf-8") as f:
            for rec in normalized_records:
                f.write(json.dumps(rec.model_dump(), ensure_ascii=False) + "\n")

        # Generate machine-readable ingestion report
        report = {
            "source": "pubmedqa",
            "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_records": len(normalized_records),
            "decision_distribution": decision_counts,
            "unique_pmids": len({r.pmid for r in normalized_records if r.pmid}),
            "interim_file": str(output_jsonl),
        }

        report_path = self.interim_dir / "ingestion_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info("Saved ingestion report to %s", report_path)
        return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = PubMedQALoader()
    rep = loader.process_and_save()
    print("PubMedQA Ingestion Summary:")
    print(json.dumps(rep, indent=2))
