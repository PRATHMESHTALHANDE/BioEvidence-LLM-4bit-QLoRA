# Agent Guidelines & Master Execution Specification: BioEvidence-LLM

> **Status**: Project Initialization  
> **Master Reference**: [Plan.md](file:///c:/Users/roari/Downloads/AI%20Projects/Fine%20Tunning%20LLM/Plan.md)  
> **Target Framework**: Python 3.10/3.11 | PyTorch | Hugging Face Transformers | TRL | PEFT | Unsloth | MLflow | Gradio

---

## 1. Project Overview & Positioning

### 1.1 Name & Objective
- **Project Name:** BioEvidence-LLM
- **Primary Objective:** Build an open-source biomedical evidence-grounded language model system that accepts a **Biomedical Question + Biomedical Evidence** and produces:
  1. **Structured Answer** (Decision: `YES`, `NO`, `MAYBE` + reasoning)
  2. **Evidence-Supported Grounding** (Direct citations from the source evidence)
  3. **Preserved Uncertainty** (Explicitly captured from source study limitations)
  4. **Documented Limitations** (Sample size, trial length, population constraints)

### 1.2 Medical Safety & Ethical Boundary
- **Strict Classification:** This system is an evidence-analysis NLP tool designed exclusively for educational and biomedical research use.
- **Prohibited Use:** It is **NOT** a clinical diagnostic system, medical advice tool, treatment recommendation engine, or autonomous clinical decision-support system.
- **Mandatory Disclaimer:** Every public interface, CLI response, Gradio UI header, and Hugging Face model card must state:
  > *"This system is intended for biomedical research and educational use. It is not a substitute for professional medical advice, diagnosis, treatment, or clinical decision-making."*
- **Evidence-First UI:** Model outputs must always display supporting evidence excerpts and source citations (PMID/PMCID) alongside generated answers.

---

## 2. Mandatory Core Engineering Principles

Antigravity must adhere strictly to these principles across every task:
1. **Evidence Before Generation:** Never generate biomedical claims that cannot be traced back to the provided source text.
2. **Zero Fact Fabrication:** Never invent p-values, sample sizes, clinical outcomes, or biological mechanisms.
3. **Preserve Uncertainty:** If source evidence is inconclusive, preserve `MAYBE` or uncertain nuance; never force a definitive `YES` or `NO`.
4. **Preserve Exact Numerical Values:** All statistics, confidence intervals, dosages, and sample counts must be preserved verbatim.
5. **Full Provenance Tracking:** Every dataset record and training sample must retain full source provenance (`source`, `pmid`, `pmcid`, `license`, `url`).
6. **Strict Leakage Prevention:** Never split records by random sampling if they share a common `PMID`, `PMCID`, or source article. Enforce article-level grouping across train and test sets.
7. **No Test Contamination:** The official PubMedQA test set and held-out evaluation sets (`data/evaluation/`) must **never** be included in training data or synthetic prompt generation.
8. **No Blind Trust of LLMs:** All LLM-transformed datasets must undergo schema validation, automated heuristic checks, and human-in-the-loop review on statistically meaningful sample subsets.
9. **Deterministic Evaluation:** Performance claims must be backed by exact numbers (Accuracy, Macro F1, Precision, Recall, JSON Validity, Hallucination Rate, Numerical Consistency) on explicitly stated evaluation splits.
10. **Modular Production Code:** Core functionality belongs in `src/`, configuration in `configs/`, tests in `tests/`. Notebooks are strictly for exploratory analysis and visualization.
11. **Local-First Architecture:** Initial development and training must run locally without requiring distributed cloud infrastructure (no Kubernetes, Kafka, Pinecone, etc.).
12. **Secret Hygiene:** Never hardcode or commit API keys, Hugging Face tokens, or credentials. Use `.env` files excluded by `.gitignore`.

---

## 3. Technology Stack & Directory Specification

### 3.1 Technology Stack
- **Language & Runtime:** Python 3.10 or 3.11 (determined by local CUDA compatibility)
- **Deep Learning & Fine-Tuning:** PyTorch (CUDA-enabled), Hugging Face (`transformers`, `datasets`, `accelerate`, `peft`, `trl`), Unsloth (where compatible), BitsAndBytes
- **NLP & Embeddings:** Sentence-Transformers, Hugging Face `evaluate`, `rouge-score`, scikit-learn
- **Data & Validation:** Pandas, NumPy, PyArrow, Pydantic v2
- **Tracking & Serving:** MLflow (local sqlite/file backend), Gradio (web UI)
- **Testing & Packaging:** Pytest, `pyproject.toml`, Hugging Face Hub CLI, optional GGUF (`llama.cpp` / Ollama)

### 3.2 Canonical Directory Structure
```text
BioEvidence-LLM/
├── .agents/
│   ├── rules/
│   └── skills/
│
├── app/
│   ├── app.py
│   └── components/
│
├── configs/
│   ├── model.yaml
│   ├── dataset.yaml
│   ├── training.yaml
│   └── evaluation.yaml
│
├── data/
│   ├── raw/
│   │   ├── pubmedqa/
│   │   ├── medquad/
│   │   ├── pubmed/
│   │   └── pmc/
│   ├── interim/
│   ├── processed/
│   ├── sft/
│   └── evaluation/
│
├── docs/
│   ├── environment-audit.md
│   ├── PROJECT_STATUS.md
│   ├── task-specification.md
│   ├── human-review-rubric.md
│   ├── architecture.md
│   ├── data/
│   ├── training/
│   ├── evaluation/
│   ├── safety.md
│   └── reproducibility.md
│
├── models/
│   ├── base/
│   ├── adapters/
│   └── gguf/
│
├── notebooks/
│   ├── 01_environment_check.ipynb
│   ├── 02_pubmedqa_analysis.ipynb
│   ├── 03_medquad_analysis.ipynb
│   ├── 04_pubmed_analysis.ipynb
│   ├── 05_dataset_analysis.ipynb
│   ├── 06_quality_analysis.ipynb
│   ├── 07_baseline.ipynb
│   ├── 08_training.ipynb
│   ├── 09_evaluation.ipynb
│   └── 10_error_analysis.ipynb
│
├── outputs/
│   ├── experiments/
│   ├── evaluation/
│   └── error_analysis/
│
├── scripts/
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── dataset/
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   ├── retrieval/
│   └── utils/
│
├── tests/
├── AGENTS.md
├── Plan.md
├── pyproject.toml
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## 4. Internal Data Schemas & Canonical Contracts

### 4.1 Canonical Data Schema (`src/dataset/schema.py`)
All ingested records from PubMedQA, MedQuAD, PubMed, and PMC must normalize into this Pydantic schema:
```json
{
  "id": "bioev_pubmedqa_12345",
  "source": "pubmedqa",
  "source_id": "12345678",
  "pmid": "12345678",
  "pmcid": null,
  "task": "evidence_qa",
  "domain": "pharmacology",
  "context": "Background: ... Methods: ... Results: ... Conclusion: ...",
  "question": "Does drug X reduce mortality in patients with condition Y?",
  "answer": "The evidence indicates that drug X reduces mortality...",
  "decision": "yes",
  "evidence": [
    "Patients receiving drug X exhibited a 24% reduction in 30-day mortality (p < 0.01)."
  ],
  "uncertainty": "The study was conducted in a single geographic cohort.",
  "limitations": [
    "Small sample size (n=120)",
    "Short follow-up period (30 days)"
  ],
  "language": "en",
  "license": "CC-BY-4.0",
  "provenance": {
    "retrieval_date": "2026-09-22",
    "source_url": "https://pubmed.ncbi.nlm.nih.gov/12345678/"
  },
  "quality_score": 0.95
}
```

### 4.2 Task Taxonomy (`task` field)
1. `evidence_qa`: Question answering grounded strictly on supplied abstract/context.
2. `evidence_classification`: Classify finding as `yes`, `no`, or `maybe` based on study findings.
3. `pico_extraction`: Extract Patient/Population, Intervention, Comparison, and Outcome.
4. `evidence_summary`: Generate a factual, non-extrapolative summary of biomedical evidence.
5. `uncertainty_extraction`: Extract hedge words, sample caveats, and statistical ambiguity.
6. `limitation_extraction`: Identify author-reported and methodological study weaknesses.
7. `medical_explanation`: Transform complex technical biomedical phrasing into plain, accurate language.

---

## 5. Master Phased Execution Roadmap

Antigravity must execute each phase sequentially. Complete verification before advancing.

```text
PHASE 0 (Audit) ➔ PHASE 1 (Env) ➔ PHASE 2 (Repo) ➔ PHASE 3 (Configs) ➔ PHASE 4 (PubMedQA)
      ➔ PHASE 5 (Schema) ➔ PHASE 6 (MedQuAD) ➔ PHASE 7 (PubMed) ➔ PHASE 8 (PMC)
      ➔ PHASE 9 (Clean & Leakage) ➔ PHASE 10 (Taxonomy) ➔ PHASE 11 (Templates)
      ➔ PHASE 12 (Synthetic Pipeline) ➔ PHASE 13 (SFT v0.1: 100 ex) ➔ PHASE 14 (Rubric)
      ➔ PHASE 15 (Scale SFT) ➔ PHASE 16 (BioEvidence-Eval: 500 ex) ➔ PHASE 17 (Baseline)
      ➔ PHASE 18 (Model Selection) ➔ PHASE 19 (Training Stack) ➔ PHASE 20 (Smoke Train)
      ➔ PHASE 21 (Full Training Matrix) ➔ PHASE 22 (MLflow) ➔ PHASE 23 (Eval Engine)
      ➔ PHASE 24 (Comparison) ➔ PHASE 25 (Error Analysis) ➔ PHASE 26 (Ablations)
      ➔ PHASE 27 (Inference Engine) ➔ PHASE 28 (Gradio App) ➔ PHASE 29 (GGUF)
      ➔ PHASE 30 (Hugging Face Hub) ➔ PHASE 31 (GitHub CI) ➔ PHASE 32 (Release v1.0)
```

---

### [x] Phase 0: Hardware & Environment Audit
- [x] Run safe read-only commands to inspect OS, CPU, RAM, GPU, VRAM, CUDA runtime, Python, and disk space.
- [x] Inspect existing PyTorch, CUDA, Transformers, PEFT, TRL, and Unsloth availability.
- [x] Generate comprehensive report at [`docs/environment-audit.md`](file:///c:/Users/roari/Downloads/AI%20Projects/Fine%20Tunning%20LLM/docs/environment-audit.md).
- [x] Specify feasible base model size range (0.5B, 1.5B, 3B, 7B/8B) and quantization strategy (4-bit QLoRA vs LoRA).
- [ ] **STOP & AWAIT USER REVIEW** before creating environments or installing packages.

---

### [ ] Phase 1: Environment Setup & Verification
- [ ] Set up virtual environment (`.venv/`) with verified Python version.
- [ ] Define dependencies in `pyproject.toml` and `requirements.txt` / `requirements-dev.txt`.
- [ ] Verify environment with programmatic verification commands:
  ```bash
  python -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
  python -c "import transformers, datasets, peft, trl; print('Core HF Libraries: OK')"
  ```
- [ ] Test and document Unsloth / BitsAndBytes compatibility on the local GPU.

---

### [ ] Phase 2: Repository Scaffolding & Git Tracking
- [ ] Initialize Git repository (`git init`).
- [ ] Create repository skeleton matching Section 3.2.
- [ ] Configure robust `.gitignore` (excluding `.venv/`, `*.env`, checkpoints `models/adapters/`, raw dumps, cache).
- [ ] Initial commit: `chore: initialize BioEvidence-LLM project structure`.

---

### [ ] Phase 3: Configuration Architecture
- [ ] Create YAML configuration files:
  - `configs/model.yaml`: Base model name, revision, sequence length, dtype, quantization mode.
  - `configs/dataset.yaml`: Dataset sources, splits, tokenization limits, task weights.
  - `configs/training.yaml`: Learning rate, batch size, gradient accumulation, LoRA rank/alpha, epochs, warmup, scheduler.
  - `configs/evaluation.yaml`: Metrics list, batch size, temperature, decoding parameters.
- [ ] Implement config loaders with Pydantic validation in `src/utils/config.py`.

---

### [ ] Phase 4: Data Ingestion — PubMedQA First
- [ ] Implement `src/data/pubmedqa_loader.py` targeting the official PubMedQA source.
- [ ] Ingest into `data/raw/pubmedqa/` with raw files preserved immutably.
- [ ] Normalize records, preserving official `yes`, `no`, `maybe` labels and train/test splits.
- [ ] Generate machine-readable ingestion report in `data/interim/pubmedqa/ingestion_report.json` with label distributions and PMID coverage.
- [ ] Write unit test `tests/test_pubmedqa_loader.py` and documentation `docs/data/pubmedqa.md`.

---

### [ ] Phase 5: Pydantic Data Schema & Contract Enforcement
- [ ] Implement `src/dataset/schema.py` containing Pydantic models:
  - `BiomedicalRecord`
  - `EvidenceQARecord`
  - `PICORecord`
  - `SummaryRecord`
  - `EvaluationRecord`
- [ ] Add strict validation for required fields, nullable handling, and license/provenance tracking.
- [ ] Write schema unit tests in `tests/test_schema.py`.

---

### [ ] Phase 6: Secondary Ingestion — MedQuAD
- [ ] Implement `src/data/medquad_loader.py`.
- [ ] Ingest into `data/raw/medquad/` and normalize to `BiomedicalRecord`.
- [ ] Perform detailed source analysis (license constraints, NIH sub-organizations, question style).
- [ ] Save source analysis report at `docs/data/medquad_analysis.md` before merging into SFT candidate pools.

---

### [ ] Phase 7: PubMed Controlled Retrieval Pipeline
- [ ] Implement `src/data/pubmed_loader.py` and `src/data/pubmed_queries.py` using official NCBI Entrez APIs.
- [ ] Support parameter-driven query ingestion:
  ```bash
  python -m src.data.pubmed_loader --query "clinical trial[pt] AND outcome" --max-records 1000
  ```
- [ ] Store complete article metadata (PMID, title, abstract, authors, journal, date, MeSH terms, publication types).

---

### [ ] Phase 8: PMC Open Access Full-Text Acquisition
- [ ] Implement `src/data/pmc_loader.py` for open-access commercial and non-commercial subsets.
- [ ] Track licenses explicitly (`redistributable_text: true/false`).
- [ ] Ingest full text and section headers into `data/raw/pmc/`.

---

### [ ] Phase 9: Preprocessing, Deduplication & Leakage Prevention
- [ ] Implement modular pipeline in `src/preprocessing/`:
  - `normalize.py`: Text cleaning, unicode normalization, whitespace stripping.
  - `validate.py`: Schema conformance, non-null context checks.
  - `deduplicate.py`: Exact SHA-256 deduplication + near-duplicate cosine similarity filtering.
  - `leakage.py`: Article/PMID-level grouping ensuring all samples from a PMID stay strictly on one side of train/eval splits.
  - `quality.py`: Automated heuristic filters (minimum context length, malformed output detection).
- [ ] Generate comprehensive `outputs/data/leakage_report.json` and step-by-step filtering counts.

---

### [ ] Phase 10: Task Specification & Deterministic Prompts
- [ ] Create detailed task guide at [`docs/task-specification.md`](file:///c:/Users/roari/Downloads/AI%20Projects/Fine%20Tunning%20LLM/docs/task-specification.md).
- [ ] Implement prompt templates in `src/dataset/prompts.py`:
  - Standardized System Prompt enforcing evidence-grounding and uncertainty preservation.
  - User instruction formatting.
  - Structured JSON target formatting.

---

### [ ] Phase 11: Controlled Synthetic Transformation Pipeline
- [ ] Build pipeline transforming source evidence into structured tasks via LLM (e.g., PICO extraction, limitation extraction).
- [ ] Ensure the LLM extracts and restructures verified evidence rather than generating ungrounded facts.
- [ ] Automated validation: Check that extracted evidence strings exist verbatim in source text.

---

### [ ] Phase 12: Pilot Dataset — BioEvidence-SFT-v0.1 (100 Samples)
- [ ] Construct balanced pilot dataset of 100 high-quality samples:
  - Evidence QA (30) | Classification (20) | PICO (15) | Summarization (15) | Uncertainty (10) | Limitations (5) | Explanation (5).
- [ ] Export to `data/sft/BioEvidence-SFT-v0.1.jsonl`.

---

### [ ] Phase 13: Human Review & Quality Rubric
- [ ] Define evaluation rubric in [`docs/human-review-rubric.md`](file:///c:/Users/roari/Downloads/AI%20Projects/Fine%20Tunning%20LLM/docs/human-review-rubric.md):
  - Factually supported | Evidence grounded | Zero invented claims | Correct uncertainty | Accurate limitations | Valid JSON.
- [ ] Record rubric review ratings (`PASS` / `FAIL` / `NEEDS_REVIEW`) for all pilot samples.

---

### [ ] Phase 14: SFT Dataset Scaling
- [ ] Scale dataset systematically: 100 $\rightarrow$ 500 $\rightarrow$ 2,000 $\rightarrow$ 5,000+ records.
- [ ] Validate distribution balance across tasks, sources, and label classes (`yes`/`no`/`maybe`).

---

### [ ] Phase 15: Golden Evaluation Set — BioEvidence-Eval-v0.1
- [ ] Curate held-out 500-sample evaluation set (`data/evaluation/BioEvidence-Eval-v0.1.jsonl`).
- [ ] Verify zero PMID overlap with training data via automated leakage checks.
- [ ] Lock and version the evaluation dataset hash.

---

### [ ] Phase 16: Zero-Shot Baseline Model Evaluation
- [ ] Run selected base model on `BioEvidence-Eval-v0.1`.
- [ ] Parse JSON outputs and score baseline metrics:
  - Decision Accuracy, Macro F1, JSON Format Validity, Hallucination Rate.
- [ ] Store predictions and baseline metrics in `outputs/evaluation/base/`.

---

### [ ] Phase 17: Fine-Tuning Pipeline Implementation
- [ ] Implement training script `src/training/train.py` using Unsloth / TRL `SFTTrainer` + PEFT LoRA.
- [ ] Support checkpoint resumption, gradient accumulation, gradient checkpointing, fp16/bf16, and evaluation logging.
- [ ] Run smoke test on 100 samples to verify forward/backward passes, memory limits, and checkpoint export.

---

### [ ] Phase 18: Experiment Matrix & MLflow Tracking
- [ ] Configure local MLflow experiment tracking.
- [ ] Execute systematic experiment grid (varying LR, LoRA rank $r \in \{16, 32\}$, epochs).
- [ ] Log loss curves, hyperparameters, commit hashes, and evaluation metrics per run.
- [ ] Save best adapter weights to `models/adapters/bioevidence-lora-best/`.

---

### [ ] Phase 19: Comprehensive Evaluation & Comparative Benchmarking
- [ ] Implement `src/evaluation/evaluator.py` evaluating both base and fine-tuned models on `BioEvidence-Eval`.
- [ ] Generate comparative evaluation report in `outputs/evaluation/comparison/evaluation_report.md`:
  - Show Base vs Fine-Tuned deltas across all tasks.
- [ ] Compute factuality and evidence consistency scores.

---

### [ ] Phase 20: Detailed Error Taxonomy Analysis
- [ ] Categorize failure cases: hallucination, unsupported claim, incorrect decision, missed uncertainty, JSON parse error.
- [ ] Generate `outputs/error_analysis/error_analysis_report.md` with qualitative failure breakdowns.

---

### [ ] Phase 21: Ablation Studies
- [ ] Evaluate performance variations across training data compositions:
  - Exp 1: PubMedQA only
  - Exp 2: PubMedQA + MedQuAD
  - Exp 3: PubMedQA + MedQuAD + structured PubMed evidence
  - Exp 4: Full Multi-task BioEvidence dataset
- [ ] Document findings in `docs/training/ablation_study.md`.

---

### [ ] Phase 22: Inference Engine & Schema Repair
- [ ] Implement `src/inference/generator.py` and `src/inference/schema_parser.py`.
- [ ] Add JSON parsing resilience with automatic syntax repair for truncated responses.
- [ ] Ensure safety guardrails attach disclaimer headers to all outputs.

---

### [ ] Phase 23: Local Gradio Demonstration App
- [ ] Implement `app/app.py` offering interactive UI:
  - Input: Biomedical question + Abstract/Evidence text.
  - Output: Decision badge (`YES`, `NO`, `MAYBE`), evidence-grounded answer, highlighted supporting sentences, uncertainty, and source PMID badge.
  - Prominent educational/research disclaimer banner.

---

### [ ] Phase 24: GGUF Export & Local Offline Inference
- [ ] Merge LoRA adapter with base model if hardware allows.
- [ ] Convert model to 4-bit / 8-bit GGUF format for execution via `llama.cpp` or Ollama.
- [ ] Document local CLI execution in `docs/inference/gguf.md`.

---

### [ ] Phase 25: Hugging Face Publishing Preparation
- [ ] Prepare Model Card (`models/adapters/README.md`) with explicit hardware, hyperparameter, benchmark, limitation, and safety documentation.
- [ ] Prepare Dataset Card for SFT and Eval datasets.
- [ ] Conduct final privacy, PII, and licensing audit.

---

### [ ] Phase 26: Test Suite, CI & v1.0 Release
- [ ] Run full automated test suite with `pytest`.
- [ ] Configure GitHub Actions workflow for linting, schema validation, and unit tests.
- [ ] Finalize master `README.md`, architecture diagrams, and release tag `v1.0.0`.

---

## 6. Antigravity Operational Protocols

1. **Step-by-Step Execution:** Never execute multiple major phases simultaneously without user sign-off.
2. **Phase 0 Gate:** Never initiate model downloading or dependency installation before executing Phase 0 audit and receiving user approval.
3. **Continuous Tracking:** Update this file's checklist and [`docs/PROJECT_STATUS.md`](file:///c:/Users/roari/Downloads/AI%20Projects/Fine%20Tunning%20LLM/docs/PROJECT_STATUS.md) as milestones are reached.
4. **Quantified Reporting:** When reporting results, provide exact metrics, test set sizes, and evaluation split hashes. Never use vague phrases like "the model performs well".
5. **Preserve Checkpoints:** Never overwrite existing training checkpoints or processed datasets without explicit confirmation.
