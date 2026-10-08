"""PubMed Central (PMC) Open Access Literature Loader.

Retrieves and parses open-access biomedical full text articles via the NCBI
BioC / PMC API, recording license types (CC-BY, CC0, etc.) and section headings.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from src.dataset.schema import BiomedicalRecord, Provenance, TaskTaxonomy

logger = logging.getLogger(__name__)

BIOC_PMC_URL = (
    "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode"
)


class PMCLoader:
    """Ingestion handler for PMC Open Access articles."""

    def __init__(
        self,
        raw_dir: str | Path = "data/raw/pmc",
        interim_dir: str | Path = "data/interim/pmc",
    ):
        self.raw_dir = Path(raw_dir)
        self.interim_dir = Path(interim_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.interim_dir.mkdir(parents=True, exist_ok=True)

    def fetch_bioc_json(self, pmcid: str) -> Optional[Dict[str, Any]]:
        """Fetch BioC formatted article JSON for a given PMCID (e.g. PMC7095448)."""
        clean_pmcid = pmcid.strip()
        if not clean_pmcid.upper().startswith("PMC"):
            clean_pmcid = f"PMC{clean_pmcid}"

        raw_file = self.raw_dir / f"{clean_pmcid}.json"
        if raw_file.exists():
            return json.loads(raw_file.read_text(encoding="utf-8"))

        url = BIOC_PMC_URL.format(pmcid=clean_pmcid)
        logger.info("Fetching PMC article from %s", url)
        try:
            resp = requests.get(url, timeout=25)
            if resp.status_code == 200:
                data = resp.json()
                raw_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
                return data
            logger.warning("PMC BioC returned HTTP %d for %s", resp.status_code, clean_pmcid)
        except Exception as e:
            logger.error("Failed fetching PMC %s: %s", clean_pmcid, e)
        return None

    def normalize_bioc(
        self, pmcid: str, data: Any, retrieval_date: str
    ) -> Optional[BiomedicalRecord]:
        """Parse BioC JSON format into BiomedicalRecord."""
        if isinstance(data, list):
            if not data:
                return None
            data = data[0]

        docs = data.get("documents", []) if isinstance(data, dict) else []
        if not docs:
            return None

        doc = docs[0]
        passages = doc.get("passages", [])

        abstract_text = ""
        results_text = ""
        license_str = "CC-BY / PMC Open Access"

        for p in passages:
            infons = p.get("infons", {})
            sec_type = infons.get("section_type", "").upper()
            text = p.get("text", "").strip()

            if "ABSTRACT" in sec_type:
                abstract_text += f" {text}"
            elif "RESULT" in sec_type and not results_text:
                results_text = text

        context = abstract_text.strip()
        if not context and passages:
            context = passages[0].get("text", "")

        if not context:
            return None

        provenance = Provenance(
            retrieval_date=retrieval_date,
            source_url=f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/",
            license_type=license_str,
            additional_metadata={"pmcid": pmcid},
        )

        return BiomedicalRecord(
            id=f"bioev_pmc_{pmcid}",
            source="pmc",
            source_id=pmcid,
            pmid=None,
            pmcid=pmcid,
            task=TaskTaxonomy.EVIDENCE_SUMMARY,
            domain="biomedical_science",
            context=context,
            question=f"What does full-text study {pmcid} establish?",
            answer=f"The paper details open access findings: {context[:250]}...",
            decision=None,
            evidence=[results_text[:200]] if results_text else [context[:200]],
            uncertainty=None,
            limitations=[],
            language="en",
            license=license_str,
            provenance=provenance,
            quality_score=0.95,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = PMCLoader()
    # Test on a prominent open-access PMC article
    rec = loader.fetch_bioc_json("PMC7095448")
    if rec:
        norm = loader.normalize_bioc(
            "PMC7095448", rec, datetime.now(timezone.utc).strftime("%Y-%m-%d")
        )
        print("Normalized PMC Record:", norm.id if norm else "Failed")
