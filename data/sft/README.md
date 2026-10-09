---
language:
- en
license: mit
task_categories:
- question-answering
- text2text-generation
tags:
- biomedical
- pubmedqa
- medquad
- evidence-based-medicine
pretty_name: BioEvidence-SFT
size_categories:
- 1K<n<10K
---

# Dataset Card: BioEvidence-SFT & BioEvidence-Eval

## Dataset Summary
The **BioEvidence-SFT** and **BioEvidence-Eval** datasets are curated instruction-tuning and benchmarking corpora designed for evidence-grounded biomedical NLP tasks.

- **GitHub Repository (Code & Pipeline):** [https://github.com/PRATHMESHTALHANDE/BioEvidence-LLM-4bit-QLoRA](https://github.com/PRATHMESHTALHANDE/BioEvidence-LLM-4bit-QLoRA)
- **Model Adapter:** [https://huggingface.co/Bhupati1998/BioEvidence-LLM-1.5B](https://huggingface.co/Bhupati1998/BioEvidence-LLM-1.5B)

## Subsets & Splits
- **BioEvidence-SFT-v0.1:** 100 balanced pilot samples across all 7 taxonomy tasks (100% human-rubric validated).
- **BioEvidence-SFT-full:** 880 multi-source training samples formatted in ChatML format.
- **BioEvidence-Eval-v0.1:** 156 held-out evaluation samples with zero article/PMID overlap with training data.

## Provenance & Sources
1. **PubMedQA (MIT License):** Expert-annotated research questions with `yes`/`no`/`maybe` decisions.
2. **MedQuAD (NIH Public Domain):** 12 NIH institute consumer and clinician health resources.
3. **PubMed (NCBI E-utilities):** Controlled queries for randomized clinical trials.
4. **PMC Open Access (CC-BY):** Full-text open-access articles via the BioC API.

## Leakage Prevention
All splits are grouped strictly by PubMed ID (`PMID`) or PubMed Central ID (`PMCID`). No articles or abstracts exist in both training and evaluation subsets.
