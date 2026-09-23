# MedQuAD Dataset & Source Analysis

> **Dataset Source:** Medical Question Answering Dataset (MedQuAD)  
> **Repository:** https://github.com/abachaa/MedQuAD  
> **Origin:** 12 National Institutes of Health (NIH) sub-institutes (NCI, NIDDK, MedlinePlus, CDC, etc.)  
> **License:** Public Domain / NIH Government Works (educational and research use)  
> **Target Path:** `data/raw/medquad/`  
> **Interim Output:** `data/interim/medquad/medquad_normalized.jsonl`

---

## 1. Source Composition & Structure
MedQuAD provides patient-oriented questions paired with expert NIH clinical answers. Unlike PubMedQA's strictly binary or tertiary decision format, MedQuAD focuses on descriptive health communication:
- **Primary Task Mapping:** `medical_explanation` and `evidence_summary`
- **Question Types:** Symptoms, treatments, diagnoses, prevention, and risk factors.
- **Tone:** Patient education and general biomedical literacy.

## 2. Integration Strategy for BioEvidence-LLM
- Used to balance the training distribution with explanatory medical literacy tasks.
- Ingested records retain the NIH institute provenance and original topic focus.
- Filtered to ensure zero overlap with PubMedQA PMIDs.
