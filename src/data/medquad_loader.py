"""MedQuAD (Medical Question Answering Dataset) Loader and Normalizer.

MedQuAD contains medical questions and answers from 12 NIH institutes
(e.g., MedlinePlus, NCI, NIDDK, CDC). This loader ingests XML-based
records, extracts questions, answers, and focus concepts, and normalizes
them into canonical BiomedicalRecord instances.
"""

import json
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import requests

from src.dataset.schema import BiomedicalRecord, Provenance, TaskTaxonomy

logger = logging.getLogger(__name__)

# Curated high-yield MedQuAD sources from official repository
MEDQUAD_RAW_URLS = [
    "https://raw.githubusercontent.com/abachaa/MedQuAD/master/1_CancerGov_QA/0000001.xml",
    "https://raw.githubusercontent.com/abachaa/MedQuAD/master/1_CancerGov_QA/0000002.xml",
    "https://raw.githubusercontent.com/abachaa/MedQuAD/master/1_CancerGov_QA/0000003.xml",
    "https://raw.githubusercontent.com/abachaa/MedQuAD/master/5_NIDDK_QA/0000001.xml",
    "https://raw.githubusercontent.com/abachaa/MedQuAD/master/5_NIDDK_QA/0000002.xml",
    "https://raw.githubusercontent.com/abachaa/MedQuAD/master/5_NIDDK_QA/0000003.xml",
    "https://raw.githubusercontent.com/abachaa/MedQuAD/master/10_MPlus_Health_Topics_QA/0000001.xml",
    "https://raw.githubusercontent.com/abachaa/MedQuAD/master/10_MPlus_Health_Topics_QA/0000002.xml",
]


class MedQuADLoader:
    """Ingestion and normalization handler for MedQuAD."""

    def __init__(
        self,
        raw_dir: str | Path = "data/raw/medquad",
        interim_dir: str | Path = "data/interim/medquad",
    ):
        self.raw_dir = Path(raw_dir)
        self.interim_dir = Path(interim_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.interim_dir.mkdir(parents=True, exist_ok=True)

    def fetch_sample_files(self) -> List[Path]:
        """Download representative MedQuAD XML files if raw directory is empty."""
        existing_xmls = list(self.raw_dir.glob("**/*.xml"))
        if existing_xmls:
            logger.info(
                "Found %d existing MedQuAD XML files in %s", len(existing_xmls), self.raw_dir
            )
            return existing_xmls

        downloaded: List[Path] = []
        for url in MEDQUAD_RAW_URLS:
            fname = url.split("/")[-2] + "_" + url.split("/")[-1]
            out_path = self.raw_dir / fname
            try:
                resp = requests.get(url, timeout=20)
                if resp.status_code == 200:
                    out_path.write_text(resp.text, encoding="utf-8")
                    downloaded.append(out_path)
            except Exception as e:
                logger.warning("Could not download %s: %s", url, e)

        logger.info("Downloaded %d MedQuAD XML sample files", len(downloaded))
        return downloaded

    def parse_xml_file(self, xml_path: Path) -> List[Dict[str, Any]]:
        """Parse a MedQuAD XML file into individual Q&A pairs."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError as err:
            logger.error("Failed to parse XML %s: %s", xml_path, err)
            return []

        doc_id = root.attrib.get("id", xml_path.stem)
        source_name = root.findtext("Source", default="NIH/MedQuAD")
        focus = root.findtext("Focus", default="")

        qa_pairs = []
        qapairs_node = root.find("QAPairs")
        if qapairs_node is not None:
            for pair in qapairs_node.findall("QAPair"):
                pid = pair.attrib.get("pid", "0")
                question_node = pair.find("Question")
                question = (
                    question_node.text.strip()
                    if question_node is not None and question_node.text
                    else ""
                )
                qtype = question_node.attrib.get("qtype", "") if question_node is not None else ""
                answer_node = pair.find("Answer")
                answer = (
                    answer_node.text.strip() if answer_node is not None and answer_node.text else ""
                )

                if question and answer:
                    qa_pairs.append(
                        {
                            "doc_id": doc_id,
                            "pid": pid,
                            "source": source_name,
                            "focus": focus,
                            "question": question,
                            "qtype": qtype,
                            "answer": answer,
                        }
                    )
        return qa_pairs

    def normalize_record(self, item: Dict[str, Any], retrieval_date: str) -> BiomedicalRecord:
        """Convert a parsed MedQuAD item into canonical BiomedicalRecord."""
        rec_id = f"bioev_medquad_{item['doc_id']}_{item['pid']}"
        context = (
            f"Topic: {item['focus']}\n"
            f"NIH Source: {item['source']}\n"
            f"Clinical Information: {item['answer']}"
        )

        provenance = Provenance(
            retrieval_date=retrieval_date,
            source_url="https://github.com/abachaa/MedQuAD",
            license_type="Research / NIH Public Domain",
            additional_metadata={
                "qtype": item["qtype"],
                "focus": item["focus"],
                "nih_sub_org": item["source"],
            },
        )

        return BiomedicalRecord(
            id=rec_id,
            source="medquad",
            source_id=f"{item['doc_id']}_{item['pid']}",
            pmid=None,
            pmcid=None,
            task=TaskTaxonomy.MEDICAL_EXPLANATION,
            domain="patient_education",
            context=context,
            question=item["question"],
            answer=item["answer"],
            decision=None,
            evidence=[item["answer"][:250] + "..."]
            if len(item["answer"]) > 250
            else [item["answer"]],
            uncertainty=None,
            limitations=[],
            language="en",
            license="Public Domain / NIH",
            provenance=provenance,
            quality_score=0.90,
        )

    def process_and_save(self) -> Dict[str, Any]:
        """Fetch, parse, and normalize MedQuAD files, saving interim jsonl."""
        files = self.fetch_sample_files()
        retrieval_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        normalized: List[BiomedicalRecord] = []

        for f in files:
            pairs = self.parse_xml_file(f)
            for p in pairs:
                normalized.append(self.normalize_record(p, retrieval_date))

        out_jsonl = self.interim_dir / "medquad_normalized.jsonl"
        with open(out_jsonl, "w", encoding="utf-8") as out_f:
            for rec in normalized:
                out_f.write(json.dumps(rec.model_dump(), ensure_ascii=False) + "\n")

        report = {
            "source": "medquad",
            "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_records": len(normalized),
            "interim_file": str(out_jsonl),
        }
        report_path = self.interim_dir / "ingestion_report.json"
        with open(report_path, "w", encoding="utf-8") as rf:
            json.dump(report, rf, indent=2)

        return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = MedQuADLoader()
    summary = loader.process_and_save()
    print("MedQuAD Ingestion Summary:", summary)
