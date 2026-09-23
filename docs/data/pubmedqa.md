# PubMedQA Ingestion & Provenance Report

> **Dataset Source:** Official PubMedQA (Biomedical Question Answering Dataset)  
> **Source URL:** https://github.com/pubmedqa/pubmedqa  
> **License:** MIT License  
> **Target Path:** `data/raw/pubmedqa/ori_pqal.json`  
> **Interim Output:** `data/interim/pubmedqa/pubmedqa_normalized.jsonl`

---

## 1. Overview & Dataset Summary
The official labeled subset of PubMedQA (`ori_pqal.json`) consists of **1,000 expert-annotated** biomedical research questions accompanied by PubMed abstracts, labeled sections (`BACKGROUND`, `METHODS`, `RESULTS`, `CONCLUSIONS`), and clinician-grounded decisions (`yes`, `no`, `maybe`).

## 2. Ingestion Statistics
- **Total Ingested Records:** 1,000
- **Unique PMIDs:** 1,000 (100% PMID coverage)
- **Class Label Distribution:**
  - `yes`: 552 (55.2%)
  - `no`: 338 (33.8%)
  - `maybe`: 110 (11.0%)
  - `unknown`: 0 (0.0%)

## 3. Normalization Strategy
1. **Context Structuring:** Retains structured section labels to allow models to recognize study methods and results.
2. **Direct Evidence Citation:** Captures the conclusion/results sentence as verbatim `evidence` text.
3. **Preserved Uncertainty:** All `maybe` records explicitly document uncertainty rather than forcing a binary classification.
4. **Provenance Tracking:** Every record retains its source URL (`https://pubmed.ncbi.nlm.nih.gov/<pmid>/`), retrieval date, and MeSH terms.
