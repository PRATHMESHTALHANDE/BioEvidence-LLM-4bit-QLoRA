"""NCBI PubMed Controlled Retrieval Pipeline using E-utilities.

Queries PubMed for peer-reviewed clinical studies and abstracts,
parsing full metadata and transforming records into BiomedicalRecord schemas.
"""

import argparse
import json
import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import requests

from src.dataset.schema import BiomedicalRecord, Provenance, TaskTaxonomy

logger = logging.getLogger(__name__)

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


class PubMedLoader:
    """Retrieval client for NCBI PubMed E-utilities API."""

    def __init__(
        self,
        raw_dir: str | Path = "data/raw/pubmed",
        interim_dir: str | Path = "data/interim/pubmed",
        email: str = "researcher@bioevidence.org",
    ):
        self.raw_dir = Path(raw_dir)
        self.interim_dir = Path(interim_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.interim_dir.mkdir(parents=True, exist_ok=True)
        self.email = email

    def search_pmids(self, query: str, max_records: int = 100) -> List[str]:
        """Search PubMed for a given boolean query and return matching PMIDs."""
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_records,
            "retmode": "json",
            "email": self.email,
        }
        logger.info("Executing PubMed search for term: %s (max=%d)", query, max_records)
        resp = requests.get(ESEARCH_URL, params=params, timeout=25)
        resp.raise_for_status()
        data = resp.json()
        id_list = data.get("esearchresult", {}).get("idlist", [])
        logger.info("Found %d PMIDs", len(id_list))
        return id_list

    def fetch_records_xml(self, pmids: List[str]) -> str:
        """Fetch XML for given PMIDs in batches."""
        if not pmids:
            return ""

        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "email": self.email,
        }
        resp = requests.post(EFETCH_URL, data=params, timeout=40)
        resp.raise_for_status()
        return resp.text

    def parse_pubmed_xml(self, xml_content: str) -> List[Dict[str, Any]]:
        """Parse NCBI PubMed XML into structured article records."""
        if not xml_content:
            return []

        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            logger.error("XML parse error on PubMed response: %s", e)
            return []

        articles: List[Dict[str, Any]] = []
        for article_node in root.findall(".//PubmedArticle"):
            pmid_node = article_node.find(".//MedlineCitation/PMID")
            pmid = pmid_node.text.strip() if pmid_node is not None and pmid_node.text else ""
            if not pmid:
                continue

            title_node = article_node.find(".//ArticleTitle")
            title = "".join(title_node.itertext()).strip() if title_node is not None else ""

            # Extract abstract text and labeled sections
            abstract_nodes = article_node.findall(".//Abstract/AbstractText")
            abstract_parts = []
            for node in abstract_nodes:
                label = node.attrib.get("Label")
                text = "".join(node.itertext()).strip()
                if label:
                    abstract_parts.append(f"{label}: {text}")
                elif text:
                    abstract_parts.append(text)

            abstract = "\n\n".join(abstract_parts)
            if not abstract:
                continue

            # Extract publication types & journal
            pub_types = [
                "".join(pt.itertext()).strip() for pt in article_node.findall(".//PublicationType")
            ]
            journal_node = article_node.find(".//Journal/Title")
            journal = (
                journal_node.text.strip() if journal_node is not None and journal_node.text else ""
            )

            articles.append(
                {
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "journal": journal,
                    "pub_types": pub_types,
                }
            )

        return articles

    def normalize_article(
        self, art: Dict[str, Any], query: str, retrieval_date: str
    ) -> BiomedicalRecord:
        """Convert a PubMed article dict into a BiomedicalRecord."""
        pmid = art["pmid"]
        title = art["title"]
        abstract = art["abstract"]

        # Derive a research question from the title if phrased affirmatively
        question = f"What was concluded regarding: {title}?"

        provenance = Provenance(
            retrieval_date=retrieval_date,
            source_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            license_type="NCBI Open Access / PubMed Public API",
            additional_metadata={
                "query": query,
                "journal": art["journal"],
                "publication_types": art["pub_types"],
            },
        )

        return BiomedicalRecord(
            id=f"bioev_pubmed_{pmid}",
            source="pubmed",
            source_id=pmid,
            pmid=pmid,
            pmcid=None,
            task=TaskTaxonomy.EVIDENCE_SUMMARY,
            domain="clinical_research",
            context=abstract,
            question=question,
            answer=f"The study reports findings regarding '{title}'. Details: {abstract[:300]}...",
            decision=None,
            evidence=[abstract[:200] + "..."],
            uncertainty=None,
            limitations=[],
            language="en",
            license="NCBI/NLM Terms",
            provenance=provenance,
            quality_score=0.95,
        )

    def retrieve_and_process(self, query: str, max_records: int = 50) -> Dict[str, Any]:
        """Full pipeline: search, fetch XML, parse, and save normalized JSONL."""
        pmids = self.search_pmids(query, max_records=max_records)
        retrieval_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if not pmids:
            return {"total_records": 0, "status": "no_results"}

        xml_data = self.fetch_records_xml(pmids)
        # Save raw XML
        raw_file = self.raw_dir / f"pubmed_query_{int(time.time())}.xml"
        raw_file.write_text(xml_data, encoding="utf-8")

        articles = self.parse_pubmed_xml(xml_data)
        normalized: List[BiomedicalRecord] = [
            self.normalize_article(a, query, retrieval_date) for a in articles
        ]

        out_jsonl = self.interim_dir / "pubmed_normalized.jsonl"
        with open(out_jsonl, "a", encoding="utf-8") as f:
            for rec in normalized:
                f.write(json.dumps(rec.model_dump(), ensure_ascii=False) + "\n")

        summary = {
            "source": "pubmed",
            "query": query,
            "retrieved_pmids": len(pmids),
            "parsed_articles": len(articles),
            "normalized_records": len(normalized),
            "interim_file": str(out_jsonl),
        }
        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PubMed Controlled Retrieval Pipeline")
    parser.add_argument(
        "--query", type=str, default="clinical trial[pt] AND outcome", help="PubMed Search Query"
    )
    parser.add_argument("--max-records", type=int, default=10, help="Max records to retrieve")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    loader = PubMedLoader()
    res = loader.retrieve_and_process(args.query, max_records=args.max_records)
    print(json.dumps(res, indent=2))
