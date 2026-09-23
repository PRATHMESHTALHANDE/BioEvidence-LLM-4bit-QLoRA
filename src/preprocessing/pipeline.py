"""Master Preprocessing, Deduplication, and Leakage Prevention Pipeline."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from src.dataset.schema import BiomedicalRecord
from src.preprocessing.deduplicate import deduplicate_records
from src.preprocessing.leakage import split_records_by_pmid
from src.preprocessing.normalize import clean_text
from src.preprocessing.quality import filter_by_quality

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """End-to-end preprocessing, cleaning, and leakage prevention pipeline."""

    def __init__(
        self,
        interim_dir: str | Path = "data/interim",
        processed_dir: str | Path = "data/processed",
        outputs_dir: str | Path = "outputs/data",
    ):
        self.interim_dir = Path(interim_dir)
        self.processed_dir = Path(processed_dir)
        self.outputs_dir = Path(outputs_dir)

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def load_interim_records(self) -> List[BiomedicalRecord]:
        """Load all normalized records from interim jsonl files."""
        records: List[BiomedicalRecord] = []
        jsonl_files = list(self.interim_dir.glob("**/*_normalized.jsonl"))

        logger.info("Found %d interim normalized JSONL files", len(jsonl_files))
        for jf in jsonl_files:
            with open(jf, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        records.append(BiomedicalRecord.model_validate(data))

        logger.info("Loaded %d total records across interim files", len(records))
        return records

    def run(self, train_ratio: float = 0.85, seed: int = 42) -> Dict[str, Any]:
        """Execute full preprocessing, deduplication, and leakage-free split."""
        raw_records = self.load_interim_records()
        total_ingested = len(raw_records)

        # 1. Clean text fields
        for r in raw_records:
            r.context = clean_text(r.context)
            r.answer = clean_text(r.answer)
            if r.question:
                r.question = clean_text(r.question)

        # 2. Quality Filtering
        quality_passed, rejected_quality = filter_by_quality(raw_records)

        # 3. Deduplication
        deduped, duplicates_removed = deduplicate_records(quality_passed)

        # 4. Leakage-free split by PMID/article
        train_recs, eval_recs, split_stats = split_records_by_pmid(
            deduped, train_ratio=train_ratio, random_seed=seed
        )

        # 5. Save processed datasets
        train_file = self.processed_dir / "train.jsonl"
        eval_file = self.processed_dir / "eval.jsonl"

        with open(train_file, "w", encoding="utf-8") as f:
            for r in train_recs:
                f.write(json.dumps(r.model_dump(), ensure_ascii=False) + "\n")

        with open(eval_file, "w", encoding="utf-8") as f:
            for r in eval_recs:
                f.write(json.dumps(r.model_dump(), ensure_ascii=False) + "\n")

        # 6. Generate leakage report
        train_pmids = {r.pmid for r in train_recs if r.pmid}
        eval_pmids = {r.pmid for r in eval_recs if r.pmid}
        overlap = train_pmids.intersection(eval_pmids)

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_ingested": total_ingested,
            "quality_rejected": rejected_quality,
            "exact_duplicates_removed": duplicates_removed,
            "deduplicated_total": len(deduped),
            "train_records": len(train_recs),
            "eval_records": len(eval_recs),
            "train_unique_articles": split_stats["train_articles"],
            "eval_unique_articles": split_stats["eval_articles"],
            "article_overlap_detected": len(overlap) > 0,
            "overlapping_pmids": list(overlap),
            "train_output_path": str(train_file),
            "eval_output_path": str(eval_file),
        }

        report_file = self.outputs_dir / "leakage_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info("Saved leakage report to %s", report_file)
        return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = PreprocessingPipeline()
    res = pipeline.run()
    print("Preprocessing & Leakage Pipeline Results:")
    print(json.dumps(res, indent=2))
