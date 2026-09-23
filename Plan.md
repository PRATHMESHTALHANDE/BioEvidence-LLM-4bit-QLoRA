# BioEvidence-LLM — Complete Antigravity Execution Plan

## 0. What we are actually building

#### Project name

BioEvidence-LLM

#### Objective

Build an open-source biomedical evidence-grounded language model system that can take:

Biomedical question + biomedical evidence

and produce:

Structured answer + evidence-supported reasoning + uncertainty + limitations

#### Example:

#### Question

Does intervention X improve outcome Y in adults with condition Z?

#### Evidence

Abstract/article excerpt from a biomedical publication.

#### Model output

```json
{
  "decision": "YES",
  "answer": "The evidence suggests that intervention X improves outcome Y...",
  "evidence": [
    "The study reported a statistically significant improvement..."
  ],
  "uncertainty": "The evidence is based on a limited population...",
  "limitations": [
    "Small sample size",
    "Short follow-up period"
  ]
}
```

The model should not pretend to be a doctor.

It should explicitly be positioned as:

Biomedical research and evidence-analysis NLP system for educational/research use. Not a clinical decision-support system.

## 1. High-level architecture

The complete system should eventually look like this:

```text
                         ┌──────────────────────┐
                         │ Public Biomedical    │
                         │ Sources              │
                         │                      │
                         │ PubMedQA             │
                         │ MedQuAD              │
                         │ PubMed               │
                         │ PMC Open Access      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Raw Data Ingestion   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Data Normalization   │
                         │ & Schema Validation  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Deduplication        │
                         │ Leakage Prevention   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Evidence Extraction  │
                         │ PICO / Findings /    │
                         │ Limitations          │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Instruction Dataset  │
                         │ Generation           │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Quality Control      │
                         │ Automated + Human    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                 ┌──────────────────┴─────────────────┐
                 │                                    │
                 ▼                                    ▼
       ┌──────────────────┐                ┌──────────────────┐
       │ Training Dataset │                │ Evaluation       │
       │ BioEvidence-SFT  │                │ BioEvidence-Eval │
       └────────┬─────────┘                └────────┬─────────┘
                │                                   │
                ▼                                   │
       ┌──────────────────┐                          │
       │ Base LLM         │                          │
       │ Qwen/Gemma etc.  │                          │
       └────────┬─────────┘                          │
                │                                   │
                ▼                                   │
       ┌──────────────────┐                          │
       │ LoRA / QLoRA     │                          │
       │ Unsloth          │                          │
       └────────┬─────────┘                          │
                │                                   │
                ▼                                   │
       ┌──────────────────┐                          │
       │ Fine-tuned       │                          │
       │ BioEvidence-LLM  │                          │
       └────────┬─────────┘                          │
                │                                   │
                └────────────────┬──────────────────┘
                                 ▼
                       ┌────────────────────┐
                       │ Evaluation Engine  │
                       └─────────┬──────────┘
                                 │
                                 ▼
                       ┌────────────────────┐
                       │ Error Analysis     │
                       └─────────┬──────────┘
                                 │
                                 ▼
                       ┌────────────────────┐
                       │ Local Application  │
                       │ Gradio             │
                       └─────────┬──────────┘
                                 │
                                 ▼
                       ┌────────────────────┐
                       │ Hugging Face Hub   │
                       │ Model + Dataset    │
                       │ Demo               │
                       └────────────────────┘
```

## 2. Antigravity's role

Antigravity should be responsible for:

- project scaffolding
- environment validation
- dependency management
- data ingestion code
- dataset processing
- dataset validation
- deduplication
- train/evaluation splitting
- prompt/template creation
- baseline evaluation
- training scripts
- LoRA/QLoRA configuration
- experiment tracking
- evaluation
- error analysis
- inference
- Gradio application
- tests
- documentation
- Hugging Face packaging
- GitHub packaging

You should remain the human reviewer, especially for:

- medical-data quality
- licensing
- final dataset samples
- training configuration
- model outputs
- publication claims

Do not allow the agent to blindly publish a generated biomedical dataset.

## 3. First: configure Antigravity correctly

Antigravity supports workspace-specific rules under .agents/rules, and its documentation recommends a persistent project rule file such as GEMINI.md/AGENTS.md to establish project conventions.

Create:

```text
BioEvidence-LLM/
│
├── .agents/
│   ├── rules/
│   └── skills/
│
├── AGENTS.md
└── ...
```

I recommend using both:

AGENTS.md

and:

.agents/rules/

## 4. Give Antigravity this master project instruction

Create:

AGENTS.md

with the following content.

### BioEvidence-LLM Antigravity Project Constitution

#### BioEvidence-LLM — Engineering Constitution

##### 1. Project Objective

Build an open-source biomedical evidence-grounded language model system named BioEvidence-LLM.

The system must analyze biomedical evidence and research questions and produce structured, evidence-grounded answers with explicit uncertainty and limitations.

The system is intended for biomedical research assistance and educational NLP research.

It is NOT a clinical diagnostic system, treatment recommendation system, medical advice system, or autonomous clinical decision-support system.

##### 2. Core Principles

The following principles are mandatory:

- Evidence before generation.
- Never invent biomedical facts.
- Preserve uncertainty from source evidence.
- Preserve numerical values exactly when supported by source material.
- Do not convert MAYBE/uncertain evidence into YES/NO.
- Every generated answer must be traceable to source evidence.
- Every dataset record must retain provenance.
- Never mix evaluation data into training data.
- Prevent article-level and PMID-level leakage.
- Do not blindly trust LLM-generated training data.
- Human-review a statistically meaningful sample of generated biomedical data.
- Prefer deterministic evaluation metrics whenever possible.
- Never report model performance without specifying dataset split and evaluation methodology.
- Never claim clinical safety or clinical effectiveness based solely on benchmark performance.
- All code must be reproducible.
- All important transformations must be logged.
- All experiments must be configurable rather than hard-coded.
- Every major pipeline must have automated tests.
- Every training experiment must produce reproducible metadata.
- Never commit API keys, Hugging Face tokens, credentials, secrets, or private data.

##### 3. Technology Direction

#### Primary language:

Python 3.10/3.11 depending on validated environment compatibility.

#### Core libraries:

- PyTorch
- Transformers
- Datasets
- Accelerate
- PEFT
- TRL
- Unsloth
- Hugging Face Hub
- Sentence Transformers
- scikit-learn
- pandas
- numpy
- pyarrow
- pydantic
- tqdm
- requests
- MLflow
- Gradio
- pytest

#### Optional:

- FAISS
- evaluate
- rouge-score
- Ollama
- llama.cpp
- bitsandbytes

#### Do not introduce:

- Kubernetes
- Kafka
- Redis
- PostgreSQL
- Elasticsearch
- Pinecone
- Databricks
- Airflow
- AWS/Azure/GCP infrastructure

unless explicitly requested later.

The first version must run locally.

##### 4. Model Strategy

Use a small Hugging Face causal language model compatible with the available local GPU.

Do NOT select a model based only on popularity.

#### First inspect:

- GPU model
- GPU VRAM
- CUDA
- PyTorch
- Unsloth compatibility
- system RAM
- free disk

Then select the model.

The project must support configuration-driven model selection.

Do not hard-code one model throughout the repository.

##### 5. Training Strategy

#### Use:

supervised fine-tuning
LoRA
QLoRA where hardware requires it
Unsloth where compatible

The base model must remain unchanged.

Adapters should be saved independently.

#### Training must support:

- resume from checkpoint
- evaluation
- logging
- checkpointing
- deterministic seeds
- configurable batch size
- gradient accumulation
- gradient checkpointing
- sequence length
- learning rate
- LoRA rank
- LoRA alpha
- dropout
- epochs
- warmup
- weight decay

##### 6. Dataset Strategy

#### Primary initial sources:

- PubMedQA
- MedQuAD
- PubMed
- PMC Open Access where licensing permits

The original PubMedQA evaluation split must never be contaminated.

Article/PMID-level leakage must be prevented.

#### Dataset records must retain:

- unique ID
- source
- source ID
- PMID/PMCID where available
- license
- task
- domain
- context
- question
- answer
- decision
- evidence
- uncertainty
- limitations
- language
- quality score
- provenance metadata

##### 7. Required Tasks

#### The first version must support:

- Evidence QA
- Evidence classification
- PICO extraction
- Evidence summarization
- Uncertainty extraction
- Limitation extraction
- Technical-to-simple biomedical explanation

Future multilingual capabilities must not be implemented until the English pipeline is stable.

##### 8. Dataset Quality

#### The pipeline must perform:

- schema validation
- null validation
- type validation
- language validation
- length validation
- exact duplicate detection
- near duplicate detection
- source validation
- license validation
- PMID/article leakage detection
- malformed-answer detection
- evidence-answer consistency checks
- unsupported-claim detection
- human-review sampling

Every filtering step must produce counts.

#### Example:

raw = 100000
schema_valid = 99500
language_valid = 99000
deduplicated = 93000
leakage_removed = 91000
quality_filtered = 85000

The pipeline must preserve these statistics.

##### 9. Train/Evaluation Separation

Never randomly split records if multiple records originate from the same biomedical article.

#### Prefer:

- article-level split
- PMID-level grouping
- source-aware splitting

#### Maintain:

data/evaluation/

separately from:

data/sft/

The evaluation dataset must not be used for training or synthetic-data generation.

##### 10. Evaluation

#### Evaluate both:

base model
fine-tuned model

using the exact same held-out evaluation set.

#### Required metrics where applicable:

- accuracy
- macro F1
- precision
- recall
- exact match
- JSON validity
- hallucination rate
- evidence consistency
- uncertainty preservation
- numerical consistency
- contradiction rate

#### For summarization:

- ROUGE where meaningful
- BERTScore where useful
- structured factuality checks

LLM-as-judge may be used only as a supplementary metric.

##### 11. Error Taxonomy

Every evaluated failure should be categorized as one or more of:

- hallucination
- unsupported claim
- wrong evidence classification
- wrong answer
- evidence contradiction
- numerical error
- missing evidence
- uncertainty loss
- limitation omission
- irrelevant answer
- formatting error
- JSON parsing error
- incomplete answer

##### 12. Reproducibility

#### Every experiment must record:

- experiment ID
- date/time
- git commit
- base model
- dataset version
- dataset hash
- training configuration
- random seed
- Python version
- PyTorch version
- Transformers version
- TRL version
- PEFT version
- Unsloth version
- GPU information
- CUDA version
- training duration
- final loss
- evaluation metrics

Use MLflow locally.

##### 13. Code Quality

#### Use:

type hints
docstrings
Pydantic models where appropriate
modular functions
configuration files
logging
meaningful exceptions
unit tests
integration tests

Do not put the entire project into notebooks.

Notebooks are only for exploration and analysis.

Production code belongs under src/.

##### 14. Security

#### Never store:

- API keys
- Hugging Face tokens
- passwords
- credentials
- private documents

inside Git.

#### Use:

environment variables
local credential stores
.env files excluded by .gitignore

##### 15. Medical Safety

Every public application and model card must state:

"This system is intended for biomedical research and educational use. It is not a substitute for professional medical advice, diagnosis, treatment, or clinical decision-making."

The application must not present model output as medically authoritative.

The UI should display source evidence alongside generated output whenever possible.

##### 16. Agent Execution Protocol

#### For every major task:

- Explore the existing code.
- Identify dependencies.
- Produce an implementation plan.
- Wait for approval if the change is architectural or high-risk.
- Implement.
- Run tests.
- Run the relevant pipeline.
- Inspect outputs.
- Fix failures.
- Report files changed.
- Report commands executed.
- Report validation results.
- Report known limitations.

Never claim a task is complete if verification was not performed.

##### 17. Definition of Done

#### A feature is complete only when:

- implementation exists
- tests exist
- tests pass
- example execution succeeds
- output has been inspected
- documentation is updated
- configuration is documented
- no secrets are committed
- reproducibility information is available

##### 18. Preferred Repository Structure

```text
BioEvidence-LLM/

├── .agents/
│ ├── rules/
│ └── skills/
│
├── data/
│ ├── raw/
│ ├── interim/
│ ├── processed/
│ ├── sft/
│ └── evaluation/
│
├── notebooks/
│
├── src/
│ ├── data/
│ ├── preprocessing/
│ ├── dataset/
│ ├── training/
│ ├── evaluation/
│ ├── inference/
│ ├── retrieval/
│ └── utils/
│
├── configs/
│ ├── dataset.yaml
│ ├── training.yaml
│ ├── evaluation.yaml
│ └── model.yaml
│
├── scripts/
│
├── tests/
│
├── models/
│ ├── base/
│ ├── adapters/
│ └── gguf/
│
├── outputs/
│
├── app/
│
├── docs/
│
├── requirements.txt
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
└── AGENTS.md
```

##### 19. Agent Behavior

Do not make broad architectural changes without explaining them.

Do not replace working components unnecessarily.

Do not introduce dependencies without justification.

Do not fabricate datasets.

Do not fabricate evaluation results.

Do not fabricate medical evidence.

Do not silently change dataset splits.

Do not silently modify evaluation data.

Do not silently download large datasets.

Before expensive operations, report estimated disk/storage/compute requirements.

When an operation may take a long time, create a resumable pipeline.

Prefer incremental execution.

The project should be able to resume after interruption.

##### 20. Final Deliverables

#### The completed project should produce:

- GitHub repository
- reproducible data pipeline
- versioned SFT dataset
- versioned evaluation dataset
- fine-tuned LoRA adapter
- merged model if hardware permits
- optional GGUF model
- evaluation report
- error analysis report
- local Gradio application
- Hugging Face model repository
- Hugging Face dataset repository
- Hugging Face demo/Space
- complete README
- technical architecture documentation
- dataset card
- model card
- experiment logs
- test suite
- reproducibility instructions

## 5. Then create Antigravity skills

This is particularly useful because Antigravity supports reusable Agent Skills through .agents/skills/<skill>/SKILL.md.

Create:

.agents/skills/

with these skills:

```text
.agents/skills/
│
├── biomedical-data-engineering/
│   └── SKILL.md
│
├── dataset-quality/
│   └── SKILL.md
│
├── biomedical-evaluation/
│   └── SKILL.md
│
├── llm-finetuning/
│   └── SKILL.md
│
├── medical-safety/
│   └── SKILL.md
│
├── ml-experiment/
│   └── SKILL.md
│
└── release-management/
    └── SKILL.md
```

The important idea is that Antigravity doesn't need to remember every project-specific rule from one huge prompt.

It can load the relevant skill when working on:

- dataset
- training
- evaluation
- release

## 6. Development phases

We will divide the entire project into 14 phases.

Do not ask Antigravity to execute all 14 phases blindly.

Execute sequentially.

### PHASE 0 — Hardware and environment discovery

#### Objective

Before writing training code, establish what your computer can actually support.

#### Antigravity prompt

Give it:

#### Antigravity Phase 0 Prompt

You are working on the BioEvidence-LLM project.

Do NOT implement the application yet.

First perform a complete local environment audit.

Inspect:

- Operating system
- CPU
- RAM
- GPU
- GPU VRAM
- NVIDIA driver
- CUDA availability
- CUDA runtime version
- Python version
- pip version
- PyTorch version
- Transformers version
- Datasets version
- PEFT version
- TRL version
- Unsloth version
- bitsandbytes availability
- Hugging Face CLI availability
- Git version
- available disk space
- SSD/HDD information where available

Run safe read-only commands.

Do not install or upgrade anything yet.

Create:

docs/environment-audit.md

#### The report must contain:

- detected hardware
- detected software
- GPU capability
- estimated feasible model sizes
- estimated training constraints
- compatibility concerns
- recommended Python environment
- recommended base model size range
- recommended precision
- whether LoRA or QLoRA is required
- approximate local disk requirements

Do not assume a GPU or VRAM size.

Do not make up compatibility.

After the audit, stop and wait for review.

#### Important

Do not start fine-tuning before this phase.

### PHASE 1 — Environment setup

After reviewing the audit, ask Antigravity to create the environment.

Target:

.venv/

Install only validated versions.

The project should preferably use:

pyproject.toml

rather than an unstructured collection of packages.

Create:

requirements.txt
requirements-dev.txt

or an equivalent reproducible dependency configuration.

#### Required verification

Antigravity must execute:

```bash
python -c "import torch; print(torch.__version__)"
```

then:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

and:

```bash
python -c "import transformers, datasets, peft, trl; print('OK')"
```

and:

```bash
python -c "import unsloth; print('Unsloth OK')"
```

If any fail:

stop and fix environment before proceeding.

### PHASE 2 — Repository creation

Create Git repository.

Expected:

BioEvidence-LLM/

Initialize:

```bash
git init
```

Create initial structure.

Commit:

chore: initialize BioEvidence-LLM project structure

### PHASE 3 — Configuration system

Do this before data engineering.

Create:

```text
configs/
├── model.yaml
├── dataset.yaml
├── training.yaml
└── evaluation.yaml
```

#### Example:

```yaml
model:
  name: ...
  revision: ...
  max_seq_length: 1024
  dtype: auto
  quantization: 4bit
```

Training:

```yaml
training:
  seed: 42
  epochs: 2
  learning_rate: 0.0002
  batch_size: 1
  gradient_accumulation_steps: 8
  warmup_ratio: 0.05
  weight_decay: 0.01
```

LoRA:

```yaml
lora:
  r: 16
  alpha: 32
  dropout: 0.05
```

Do not treat these values as final.

They are experiment configuration.

### PHASE 4 — Data acquisition

Now implement:

src/data/

Create:

pubmedqa_loader.py
medquad_loader.py
pubmed_loader.py
pmc_loader.py
source_registry.py

The pipeline should produce:

```text
data/raw/
├── pubmedqa/
├── medquad/
├── pubmed/
└── pmc/
```

## 7. PubMedQA first

Start with PubMedQA.

The official repository provides the dataset and evaluation procedure and explicitly uses the labels yes, no, and maybe.

Do not begin with all biomedical sources simultaneously.

First get:

PubMedQA → normalized dataset → validation → evaluation

working.

#### Antigravity task

### Phase 4A — PubMedQA Ingestion

Implement the PubMedQA ingestion pipeline.

#### Requirements:

- Download/use the official PubMedQA source.
- Preserve original files.
- Do not modify raw data.
- Store metadata about source and retrieval.
- Parse the official dataset.
- Normalize records into our internal schema.
- Preserve PMID/article identifiers.
- Preserve the original yes/no/maybe label.
- Preserve train/test separation.
- Never include official test examples in SFT training data.
- Validate every record.
- Produce ingestion statistics.

Create:

src/data/pubmedqa_loader.py

tests/test_pubmedqa_loader.py

docs/data/pubmedqa.md

data/interim/pubmedqa/

Create a machine-readable ingestion report.

#### Required report:

- total records
- train records
- validation records if applicable
- test records
- missing fields
- malformed records
- label distribution
- duplicate records
- PMID coverage

Run tests and show results.

Do not continue to MedQuAD until PubMedQA ingestion passes validation.

### PHASE 5 — Internal data schema

Create Pydantic models.

For example:

src/dataset/schema.py

#### Models:

- BiomedicalRecord
- EvidenceQARecord
- PICORecord
- SummaryRecord
- EvaluationRecord

#### The canonical record should contain:

```json
{
  "id": "...",
  "source": "...",
  "source_id": "...",
  "pmid": "...",
  "pmcid": "...",
  "task": "...",
  "domain": "...",
  "context": "...",
  "question": "...",
  "answer": "...",
  "decision": "...",
  "evidence": [],
  "uncertainty": "...",
  "limitations": [],
  "language": "en",
  "license": "...",
  "provenance": {},
  "quality_score": null
}
```

Make fields nullable where appropriate.

Do not force fake values.

### PHASE 6 — MedQuAD

After PubMedQA works:

```text
MedQuAD
   ↓
loader
   ↓
normalization
   ↓
schema validation
   ↓
provenance
```

Do not immediately merge it into the SFT dataset.

First perform:

source analysis

#### Questions Antigravity must answer:

- What type of questions?
- What organizations/sites?
- What licenses?
- How much duplication?
- What style?
- What medical domains?
- Does it contain answers that are too generic?
- Does it complement PubMedQA?

Then decide how much to use.

### PHASE 7 — PubMed acquisition

Build a controlled PubMed retrieval pipeline.

Do not scrape randomly.

Use official APIs/interfaces.

Create:

src/data/pubmed_loader.py
src/data/pubmed_queries.py

The pipeline should support:

```bash
python -m src.data.pubmed_loader \
    --query "..." \
    --max-records 1000
```

Every downloaded record should preserve:

- PMID
- title
- abstract
- authors
- journal
- publication date
- MeSH terms where available
- publication types
- source URL
- retrieval timestamp

### PHASE 8 — PMC

Only after PubMed works.

#### For PMC:

- PMCID
- article metadata
- full text
- license
- source URL
- retrieval timestamp

#### Critical:

Do not assume every PMC article can be redistributed.

The dataset pipeline must track licensing.

For example:

```json
{
  "license": "CC BY",
  "redistributable_text": true
}
```

or:

```json
{
  "license": "unknown",
  "redistributable_text": false
}
```

This becomes extremely important when publishing to Hugging Face.

### PHASE 9 — Data cleaning

Now create:

```text
src/preprocessing/
├── normalize.py
├── validate.py
├── deduplicate.py
├── leakage.py
├── language.py
├── pii.py
└── quality.py
```

#### Pipeline:

```text
raw
 ↓
normalize
 ↓
schema validation
 ↓
language filter
 ↓
length filter
 ↓
duplicate removal
 ↓
article-level grouping
 ↓
leakage detection
 ↓
quality filtering
 ↓
processed
```

## 10. Deduplication

Implement two levels.

#### Level 1 — Exact

Hash:

question
context
answer

using SHA-256.

#### Level 2 — Near duplicate

Use embeddings.

For example:

sentence-transformers

then:

FAISS

or another local similarity mechanism.

But don't immediately remove everything above a threshold.

Instead:

candidate duplicate

→ review logic

→ remove only when justified.

Create:

docs/data/deduplication.md

with the chosen threshold and rationale.

## 11. Leakage prevention

This is one of the most important parts.

Suppose:

Article A

has:

question 1
question 2
question 3

You cannot have:

question 1 → train
question 2 → test

because the model has effectively seen the article.

Instead:

Article A → train
Article B → train
Article C → test

Implement:

src/preprocessing/leakage.py

Group by:

PMID
PMCID
source_id

where available.

Generate:

leakage_report.json

### PHASE 10 — Create the task taxonomy

Define:

task

with controlled values:

evidence_qa
evidence_classification
pico_extraction
evidence_summary
uncertainty_extraction
limitation_extraction
medical_explanation

Create:

docs/task-specification.md

For each task specify:

#### Input

Output
Required fields
Allowed values
Evaluation metric
Failure modes
Example

### PHASE 11 — Create instruction templates

Do not let an LLM randomly generate instructions.

Create deterministic templates.

#### Example:

#### SYSTEM:

You are a biomedical evidence analysis assistant.

Use only the provided evidence.

Do not introduce information that is absent from the evidence.

Preserve uncertainty.

#### USER:

Question:
{question}

Evidence:
{context}

#### Return JSON:

```json
{
  "decision": "yes|no|maybe",
  "answer": "...",
  "evidence": [],
  "uncertainty": "...",
  "limitations": []
}
```

Create:

src/dataset/prompts.py

## 12. Synthetic transformation

This is where an LLM may be used.

But the workflow must be:

```text
trusted biomedical source
       ↓
structured extraction
       ↓
LLM transformation
       ↓
quality validation
       ↓
human review
       ↓
SFT dataset
```

Not:

```text
LLM
 ↓
medical facts
```

The LLM should transform source information, not invent the medical truth.

## 13. Build 100 examples first

Do not generate 30,000 records immediately.

Build:

BioEvidence-SFT-v0.1

with approximately:

100 examples

distributed across tasks.

For example:

Evidence QA            30
Classification         20
PICO                   15
Summarization          15
Uncertainty            10
Limitations            5
Explanation            5

Then manually inspect all 100.

## 14. Human review

Create:

docs/human-review-rubric.md

#### Score:

1. Factually supported
2. Evidence grounded
3. No invented claims
4. Correct classification
5. Correct uncertainty
6. Correct limitations
7. Clear answer
8. Correct structure

#### Use:

PASS
FAIL
NEEDS_REVIEW

Do not use a subjective "looks good" rating.

## 15. Scale dataset

Only after 100 examples pass.

Scale:

```text
100
 ↓
500
 ↓
2,000
 ↓
5,000
 ↓
10,000+
```

The final number should depend on:

data quality
hardware
task balance
training behavior
evaluation results

Not on an arbitrary target.

## 16. Build BioEvidence-Eval

This is separate from SFT.

Create:

data/evaluation/

with:

BioEvidence-Eval-v0.1

#### Initial target:

500 examples

#### Example:

100 Evidence QA
100 Evidence Classification
100 PICO
100 Summarization
100 Uncertainty/Limitation

But the actual distribution should be determined from available validated examples.

## 17. Baseline model

Before fine-tuning:

```text
Base Model
   ↓
BioEvidence-Eval
   ↓
baseline_predictions.jsonl
   ↓
metrics
```

This gives us:

What does the model already know?

Then:

```text
Fine-tuned Model
   ↓
same BioEvidence-Eval
```

Now we can compare.

## 18. Model selection

Only now select the actual base model.

The current Hugging Face ecosystem supports SFT with PEFT/LoRA, and TRL's SFTTrainer supports adapter training.

But the exact model should depend on your hardware.

Antigravity should inspect the Phase 0 audit and then propose:

Candidate A
Candidate B
Candidate C

For each:

- parameter count
- VRAM requirement
- context length
- license
- Unsloth compatibility
- quantization availability
- expected training feasibility

Then select one based on your hardware, rather than blindly choosing a 7B/8B model.

## 19. Training architecture

#### Use:

Unsloth
+
Transformers
+
TRL
+
PEFT

TRL's current SFT tooling supports conversational and prompt-completion datasets and PEFT adapters.

#### Architecture:

```text
Base Model
     │
     ▼
4-bit quantization if required
     │
     ▼
LoRA adapters
     │
     ▼
SFTTrainer / Unsloth
     │
     ▼
Adapter checkpoint
```

## 20. First training run

Do NOT train the full dataset first.

#### Use:

100–200 examples

for a smoke test.

Objective:

Does training work?

Not:

Is the model excellent?

Verify:

- model loads
- tokenizer loads
- dataset loads
- batches work
- forward pass works
- backward pass works
- loss decreases
- checkpoint saves
- adapter saves
- inference works

## 21. Training configuration

Create:

configs/training.yaml

with parameters such as:

```yaml
seed: 42

training:
  num_train_epochs: 1
  learning_rate: ...
  per_device_train_batch_size: 1
  gradient_accumulation_steps: ...
  warmup_ratio: 0.05
  weight_decay: 0.01

lora:
  r: 16
  alpha: 32
  dropout: 0.05

optimization:
  gradient_checkpointing: true
  fp16: true
  bf16: false

logging:
  mlflow: true

checkpointing:
  save_strategy: steps
```

These are starting values, not universal final values.

## 22. Experiment matrix

Don't perform random tuning.

Start with:

#### Experiment A

LR: 1e-4
LoRA r: 16
epochs: 1

#### Experiment B

LR: 2e-4
LoRA r: 16
epochs: 1

#### Experiment C

LR: 2e-4
LoRA r: 32
epochs: 1

#### Experiment D

LR: 1e-4
LoRA r: 32
epochs: 2

But only after a successful smoke test.

## 23. MLflow

Track every run.

#### Example:

Experiment:
BioEvidence-SFT

#### Run:

exp_001
exp_002
exp_003

Record:

- dataset_version
- model
- learning_rate
- epochs
- LoRA rank
- LoRA alpha
- sequence length
- batch size
- GPU
- training time
- train loss
- eval loss
- metrics

## 24. Evaluation engine

Create:

```text
src/evaluation/
├── evaluator.py
├── metrics.py
├── classification.py
├── structured.py
├── factuality.py
├── hallucination.py
└── reports.py
```

#### Run:

```text
Base model
     ↓
evaluation
     ↓
base_results.json

Fine-tuned model
     ↓
evaluation
     ↓
finetuned_results.json
```

## 25. Compare models

Create:

```text
outputs/evaluation/
├── base/
├── finetuned/
└── comparison/
```

Generate:

evaluation_report.json
evaluation_report.md

#### Comparison should include:

- metric
- base
- fine-tuned
- delta

Do not create an overall "winner" score.

Instead report task-specific results.

## 26. Error analysis

This is extremely important for your resume/research credibility.

Create:

outputs/error_analysis/

#### Analyze:

- wrong answer
- hallucination
- unsupported claim
- wrong evidence
- numerical error
- uncertainty loss
- limitation loss
- format error

Then generate:

error_analysis_report.md

with examples.

## 27. Ablation studies

After the core model works, perform controlled experiments.

For example:

Base model

vs

Base + PubMedQA

vs

Base + PubMedQA + MedQuAD

vs

Base + multi-task dataset

This gives you a meaningful research story.

## 28. Important experiment

A particularly useful experiment:

#### Training data composition

#### Experiment 1

PubMedQA only

#### Experiment 2

PubMedQA + MedQuAD

#### Experiment 3

PubMedQA + MedQuAD + structured PubMed evidence

#### Experiment 4

Multi-task BioEvidence dataset

Then evaluate all on the same held-out evaluation dataset.

## 29. Build inference engine

Create:

```text
src/inference/
├── loader.py
├── generator.py
├── schema_parser.py
└── safety.py
```

The inference engine should:

question
+

```text
evidence
     ↓
model
     ↓
raw response
     ↓
JSON parser
     ↓
validation
     ↓
structured response
```

If JSON is invalid:

repair/retry

but do not silently invent fields.

## 30. Build local application

#### Use:

Gradio

#### Application:

app/app.py

#### UI:

```text
Biomedical Question
──────────────────────
[ Enter question ]

Evidence
──────────────────────
[ Paste abstract/article ]

[ Analyze Evidence ]

Result
──────────────────────

Decision:
MAYBE

Answer:
...

Supporting Evidence:
...

Uncertainty:
...

Limitations:
...
```

Add:

⚠ Research/Educational Use Only
Not medical advice.

## 31. Add evidence highlighting

If possible:

```text
Model claim
     ↓
supporting evidence sentence
```

For example:

Claim:
Treatment X improved outcome Y.

Supporting evidence:
"The intervention group demonstrated..."

This is much more useful than simply showing generated text.

## 32. Add citation/provenance

Every input should preserve:

source
PMID
PMCID
title
URL
retrieval date

Then the UI can show:

#### Source:

PubMed

#### PMID:

12345678

## 33. Local GGUF

Only after the normal Transformers model works.

#### Pipeline:

```text
Fine-tuned model
      ↓
merge adapter if appropriate
      ↓
convert
      ↓
GGUF
      ↓
llama.cpp / Ollama
```

Do not make GGUF the primary format.

Primary:

Hugging Face Transformers

Secondary:

GGUF

## 34. Hugging Face publishing structure

Eventually create:

username/BioEvidence-SFT
username/BioEvidence-Eval
username/BioEvidence-Model
username/BioEvidence-GGUF

Potentially:

username/BioEvidence-Demo

for the application.

Before publication, perform:

license audit
dataset provenance audit
PII audit
test-set leakage audit
model card review
dataset card review

## 35. Model card

#### The model card must document:

- Model
- Base model
- Fine-tuning method
- Dataset
- Dataset versions
- Training hardware
- Training parameters
- Evaluation
- Limitations
- Known failure modes
- Intended use
- Out-of-scope use
- Safety considerations
- License

Never claim:

medical-grade
doctor-level
clinically validated
safe for patient diagnosis

unless you actually have appropriate evidence.

## 36. Dataset card

#### Document:

- Dataset sources
- Source licenses
- Transformation process
- Filtering
- Deduplication
- Leakage prevention
- Human review
- Known biases
- Known limitations
- Intended use
- Prohibited use
- Version
- Statistics

## 37. GitHub README

#### README should contain:

- BioEvidence-LLM

- Overview

- Architecture

- Capabilities

- Dataset

- Training

- Evaluation

- Installation

- Quickstart

- Example

- Results

- Limitations

- Safety

- Reproducibility

- Project Structure

- Contributing

- License

## 38. Automated tests

#### Minimum:

```text
tests/
├── test_schema.py
├── test_data_loader.py
├── test_preprocessing.py
├── test_deduplication.py
├── test_leakage.py
├── test_prompting.py
├── test_inference.py
├── test_evaluation.py
└── test_app.py
```

#### Run:

pytest

before every major release.

## 39. CI

Eventually GitHub Actions:

```text
push
 ↓
lint
 ↓
unit tests
 ↓
schema tests
 ↓
small integration test
```

Do not run expensive model training in CI.

## 40. Final project structure

The final structure should resemble:

```text
BioEvidence-LLM/
│
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
│   │
│   ├── interim/
│   ├── processed/
│   ├── sft/
│   └── evaluation/
│
├── docs/
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
│
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
│
├── AGENTS.md
├── pyproject.toml
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## 41. How YOU should operate Antigravity

This is important.

Don't continuously say:

"Build everything."

Instead:

Conversation 1
Perform Phase 0.
Do not modify anything.

Review.

Conversation 2
Execute Phase 1.

Review.

Conversation 3
Execute Phase 2 and Phase 3.
Run all tests.

Review.

Conversation 4
Execute Phase 4A: PubMedQA ingestion.
Stop when validation is complete.

Review the actual dataset.

Conversation 5
Execute MedQuAD ingestion.
Do not merge into SFT yet.
Generate source analysis.
Conversation 6
Execute PubMed ingestion.

And so on.

This gives you control.

Antigravity itself recommends separating exploration, planning and execution, and establishing local verification loops rather than letting an autonomous agent make unverified changes.

## 42. Use /goal selectively

Antigravity provides /goal for tasks intended to run through to completion.

For example:

/goal Implement the PubMedQA ingestion pipeline, run all unit tests, execute the pipeline on a small sample, inspect the generated schema, and produce a validation report.

This is appropriate for relatively bounded tasks.

Do not use:

/goal Build the entire BioEvidence-LLM system.

That is too broad.

## 43. Use implementation plans

For major phases, ask:

Explore the repository and create an implementation plan for Phase 6.
Do not modify files yet.

Antigravity will create an implementation-plan artifact that you can review before proceeding.

Then:

Proceed with the approved implementation plan.
Run all relevant tests.

## 44. Use separate agents where useful

Because Antigravity supports parallel agents, you can eventually divide work.

For example:

Agent A
Dataset engineering
Agent B
Evaluation framework
Agent C
Application/UI
Agent D
Documentation

But don't do this at the beginning.

First establish the core architecture.

## 45. Create an Antigravity project dashboard

Ask Antigravity to create:

docs/PROJECT_STATUS.md

#### Example:

# BioEvidence-LLM Project Status

## Phase 0
- [x] Environment audit

## Phase 1
- [x] Python environment
- [x] Dependencies
- [x] GPU validation

## Phase 2
- [x] Repository

## Phase 3
- [x] Configuration

## Phase 4
- [x] PubMedQA
- [ ] MedQuAD
- [ ] PubMed
- [ ] PMC

## Phase 5
- [ ] Data normalization

## Phase 6
- [ ] Dataset generation

## Phase 7
- [ ] Human validation

## Phase 8
- [ ] Evaluation dataset

## Phase 9
- [ ] Baseline

## Phase 10
- [ ] Fine-tuning

## Phase 11
- [ ] Evaluation

## Phase 12
- [ ] Application

## Phase 13
- [ ] Hugging Face

## Phase 14
- [ ] Release

Every completed phase should update this.

## 46. Create an experiment registry

Create:

experiments/

and:

experiments/README.md

#### Each experiment:

EXP-001
EXP-002
EXP-003

#### Example:

```yaml
experiment_id: EXP-003
model: ...
dataset: BioEvidence-SFT-v0.2
dataset_hash: ...
learning_rate: ...
epochs: ...
lora_r: ...
seed: 42
gpu: ...
```

This will make the project much more research-grade.

## 47. The most important development rule

Do not allow Antigravity to say:

"The model performs well."

Instead it must say:

Evidence QA accuracy: 78.4%
Macro F1: 75.2%
JSON validity: 98.1%
Hallucination rate: 7.3%

and specify:

Dataset:
BioEvidence-Eval-v0.2

N:
500

Split:
held-out article-level test set

This makes the project scientifically defensible.

## 48. What the final application should demonstrate

#### The demo should allow:

```text
Input
Question:

Does metformin reduce cardiovascular risk in patients
with type 2 diabetes?
Evidence
Paste biomedical abstract...
Output
Evidence Assessment
────────────────────────────

Decision:
MAYBE

Evidence-grounded answer:
...

Supporting evidence:
1. ...
2. ...

Uncertainty:
...

Limitations:
...

Source:
PubMed
PMID: XXXXXXXX
```

## 49. What makes this project strong technically

The project should demonstrate more than fine-tuning.

#### Your architecture demonstrates:

Data Engineering
       +
Biomedical NLP
       +
RAG/evidence grounding concepts
       +
LLM Fine-tuning
       +
LoRA/QLoRA
       +
Unsloth
       +
Evaluation
       +
Experiment Tracking
       +
Hallucination Analysis
       +
Model Deployment
       +
Hugging Face
       +
Responsible AI

That is much stronger than:

"I fine-tuned a 7B model on PubMedQA."

## 50. One important correction to the original plan

I would not make the first version a RAG system plus fine-tuning simultaneously.

First build:

```text
Evidence
 ↓
Fine-tuned model
 ↓
Structured answer
```

Then add:

```text
PubMed/PMC retrieval
 ↓
Relevant evidence
 ↓
Fine-tuned model
 ↓
Answer
```

So the project evolves:

V0.1
Dataset
V0.2
Fine-tuned model
V0.3
Evaluation framework
V0.4
Local application
V0.5
Biomedical retrieval
V1.0
Retrieval
+
Evidence-grounded fine-tuned model
+
Evaluation
+
UI

This separation makes it much easier to determine whether improvements come from retrieval or fine-tuning.

## 51. Final V1 architecture

Eventually:

```text
                    USER
                      │
                      ▼
             ┌─────────────────┐
             │ Research        │
             │ Question        │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ PubMed / PMC    │
             │ Retriever       │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Evidence        │
             │ Selection       │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ BioEvidence-LLM │
             │ LoRA/QLoRA      │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Structured      │
             │ Output          │
             └────────┬────────┘
                      │
          ┌───────────┼────────────┐
          ▼           ▼            ▼
       Answer      Evidence     Uncertainty
          │           │            │
          └───────────┼────────────┘
                      ▼
              ┌──────────────┐
              │ Provenance   │
              │ / Citations  │
              └──────────────┘
```

## 52. The exact first prompt I recommend you give Antigravity

Don't give it the entire implementation request immediately.

Start with this:

#### Initial Antigravity Project Prompt

I want you to act as the principal ML engineer for the BioEvidence-LLM project.

First read AGENTS.md completely and follow all project rules.

Do NOT start implementing the entire system.

We will build the project incrementally using verified phases.

Your first task is PHASE 0 ONLY.

Perform a complete read-only audit of the local machine and current development environment.

Inspect:

- operating system
- CPU
- RAM
- GPU
- VRAM
- NVIDIA driver
- CUDA
- Python
- PyTorch
- Transformers
- Datasets
- PEFT
- TRL
- Unsloth
- bitsandbytes
- Hugging Face CLI
- Git
- available disk space

Determine what model sizes are realistically feasible for local LoRA/QLoRA training.

Do not install, upgrade, uninstall, download large datasets, or modify project files except for the Phase 0 audit report.

Create:

docs/environment-audit.md

The report must include:

- Hardware
- Software
- GPU/VRAM
- CUDA/PyTorch status
- Unsloth compatibility observations
- Recommended Python environment
- Recommended model size range
- Recommended quantization strategy
- Recommended sequence length range
- Estimated storage requirements
- Potential risks
- Recommended next step

Do not guess missing information.

Run safe commands to verify everything.

At the end, provide:

- commands executed
- files created
- results
- blockers
- recommendation

STOP after Phase 0.
Do not proceed to Phase 1.

## 53. Then we work together phase-by-phase

#### The overall sequence I recommend is:

```text
PHASE 0
Environment audit
        ↓
PHASE 1
Environment setup
        ↓
PHASE 2
Repository
        ↓
PHASE 3
Configuration
        ↓
PHASE 4
PubMedQA
        ↓
PHASE 5
MedQuAD
        ↓
PHASE 6
PubMed
        ↓
PHASE 7
PMC
        ↓
PHASE 8
Normalization
        ↓
PHASE 9
Deduplication + leakage
        ↓
PHASE 10
Task generation
        ↓
PHASE 11
Human validation
        ↓
PHASE 12
BioEvidence-SFT
        ↓
PHASE 13
BioEvidence-Eval
        ↓
PHASE 14
Baseline
        ↓
PHASE 15
Fine-tuning
        ↓
PHASE 16
Evaluation
        ↓
PHASE 17
Error analysis
        ↓
PHASE 18
Ablation
        ↓
PHASE 19
Inference
        ↓
PHASE 20
Gradio
        ↓
PHASE 21
Retrieval
        ↓
PHASE 22
GGUF/Ollama
        ↓
PHASE 23
Hugging Face
        ↓
PHASE 24
GitHub
        ↓
PHASE 25
Documentation
        ↓
PHASE 26
Release v1.0
```

#### One thing I would do differently from a normal coding project

Don't let Antigravity make the hardware/model decision itself before we see your machine.

The first thing to give it is the Phase 0 prompt above. Once it produces docs/environment-audit.md, bring that result here.

From that audit, we can lock down:

- exact base model
- exact quantization
- exact Unsloth configuration
- maximum sequence length
- batch size
- gradient accumulation
- expected training time
- dataset size we should initially target
- whether 0.5B / 1.5B / larger is realistic
- exact package versions

Then Antigravity can execute the rest of the plan against your actual hardware, rather than us designing an impressive-looking pipeline that your local machine cannot train.

The current Hugging Face/TRL stack supports SFT with PEFT adapters, including LoRA/QLoRA workflows, while the official PubMedQA repository provides the benchmark data and its established evaluation format.