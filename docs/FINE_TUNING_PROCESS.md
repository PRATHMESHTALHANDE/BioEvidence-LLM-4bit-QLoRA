# BioEvidence-LLM: Fine-Tuning & Model Architecture Master Reference

> **Document Type:** Master Reference & Reproducibility Guide  
> **Target Base Model:** `Qwen/Qwen2.5-1.5B-Instruct`  
> **Target Quantization:** 4-bit NormalFloat4 (NF4) via BitsAndBytes  
> **Tuning Method:** Parameter-Efficient Fine-Tuning (PEFT / QLoRA)  
> **Primary Tracking Engine:** Local SQLite MLflow (`outputs/experiments/mlflow.db`)

---

## 1. System Objective & Safety Scope

BioEvidence-LLM fine-tunes modern lightweight open-weight language models to perform **evidence-grounded scientific literature analysis**. Given a biomedical question and an abstract/study context, it outputs structured JSON containing a decision (`YES`, `NO`, `MAYBE`), cited verbatim excerpts, preserved statistical uncertainty, and clinical trial limitations.

> ⚠️ **MANDATORY SAFETY BOUNDARY:**  
> This system is designed exclusively for biomedical literature analysis and educational/scientific research. It is **not** a clinical diagnostic system or medical provider substitute.

---

## 2. Hardware Profile & Memory Constraints

- **Detected GPU:** NVIDIA GeForce RTX 3050 Laptop GPU
- **VRAM Available:** **4,096 MiB (4.0 GB GDDR6)**
- **CUDA Version:** 12.4 / 12.6 supported
- **System Memory:** 16 GB RAM, 10-core Intel Core i7-12650H

### Why 7B/8B Models Were Ruled Out Locally
A standard 7B or 8B parameter model requires ~4.5 GB of VRAM just to store 4-bit quantized base weights, exceeding 6 GB during backward passes. Training on 4 GB VRAM would cause immediate CUDA Out-Of-Memory (OOM) failures.

### The 1.5B QLoRA Sweet Spot
`Qwen/Qwen2.5-1.5B-Instruct` occupies only **~1.1 GB VRAM** in 4-bit NF4 precision. Adding trainable LoRA matrices, 8-bit optimizer states, and gradient checkpointing brings total VRAM consumption during training to **~2.6–3.0 GB**, safely within the 4.0 GB limit while supporting sequence lengths up to 2048 tokens.

---

## 3. Fine-Tuning Hyperparameters & Architecture

All training parameters are configured in [`configs/training.yaml`](file:///c:/Users/roari/Downloads/AI%20Projects/Fine%20Tunning%20LLM/configs/training.yaml) and [`configs/model.yaml`](file:///c:/Users/roari/Downloads/AI%20Projects/Fine%20Tunning%20LLM/configs/model.yaml):

| Hyperparameter | Configuration Value | Technical Rationale |
|---|---|---|
| **Base Model** | `Qwen/Qwen2.5-1.5B-Instruct` | State-of-the-art instruction following and reasoning. |
| **Quantization** | 4-bit NormalFloat4 (`nf4`), double quantization | Compresses 16-bit weights to 4-bit with minimal perplexity degradation. |
| **Compute Dtype** | `torch.float16` | Accelerated tensor computation on RTX 3050 Tensor Cores. |
| **PEFT Method** | LoRA (`LoraConfig`) | Freezes base model; trains only low-rank adapter matrices. |
| **LoRA Rank ($r$)** | `16` | Balances expressive capacity with memory footprint. |
| **LoRA Alpha ($\alpha$)** | `32` | Standard $2\times$ rank scaling factor for stable updates. |
| **Target Modules** | All 7 linear projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) | Adapts attention heads and feed-forward networks simultaneously. |
| **Trainable Parameters** | **18,464,768** (~1.18% of total 1.56B parameters) | Fast convergence and low memory overhead. |
| **Optimizer** | `paged_adamw_8bit` | Offloads inactive optimizer pages to system RAM to prevent VRAM spikes. |
| **Batch Size** | 1 per device | Minimizes activation tensor memory. |
| **Gradient Accumulation** | 8 steps | Effective batch size of $1 \times 8 = 8$. |
| **Learning Rate** | $2 \times 10^{-4}$ ($0.0002$) | Standard stable rate for LoRA fine-tuning. |
| **LR Scheduler** | Cosine with 5% warmup | Prevents gradient shock in early steps. |
| **Gradient Checkpointing** | Enabled | Trades ~20% compute time for ~60% activation memory savings. |

---

## 4. Dataset Pipeline & Provenance

Training and benchmark datasets are built from verified biomedical literature:

1. **PubMedQA (`data/raw/pubmedqa/ori_pqal.json`):** 1,000 expert-labeled clinical studies.
2. **MedQuAD (`data/raw/medquad/`):** 27 curated NIH patient-education and clinical Q&A pairs.
3. **NCBI PubMed API (`data/raw/pubmed/`):** Controlled clinical trial queries.
4. **PMC Open Access (`data/raw/pmc/`):** Full-text BioC articles with section headings.

### Leakage Prevention Guarantee
To ensure zero evaluation contamination, records are grouped strictly by PubMed ID (`PMID`):
* **Training Set ([`data/sft/BioEvidence-SFT-full.jsonl`](file:///c:/Users/roari/Downloads/AI%20Projects/Fine%20Tunning%20LLM/data/sft/BioEvidence-SFT-full.jsonl)):** 880 unique articles.
* **Held-Out Benchmark ([`data/evaluation/BioEvidence-Eval-v0.1.jsonl`](file:///c:/Users/roari/Downloads/AI%20Projects/Fine%20Tunning%20LLM/data/evaluation/BioEvidence-Eval-v0.1.jsonl)):** 156 unique articles.
* **Article Overlap:** **0 PMIDs** (confirmed in [`outputs/data/leakage_report.json`](file:///c:/Users/roari/Downloads/AI%20Projects/Fine%20Tunning%20LLM/outputs/data/leakage_report.json)).

---

## 5. Execution Commands & Workflow

### 5.1 Verification Smoke Test
Verifies end-to-end forward pass, backpropagation, and adapter export:
```powershell
.\.venv\Scripts\python.exe -m src.training.train --smoke-test
```

### 5.2 Full Fine-Tuning Run
Executes the full 3-epoch training across the complete instruction dataset:
```powershell
.\.venv\Scripts\python.exe -m src.training.train --dataset data/sft/BioEvidence-SFT-full.jsonl
```

### 5.3 Launch Training Metrics Dashboard (MLflow)
Opens interactive loss and hyperparameter tracking in your browser:
```powershell
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///outputs/experiments/mlflow.db
```
Navigate to: `http://127.0.0.1:5000`

### 5.4 Launch Interactive Clinical Demonstration (Gradio UI)
Launches the browser web demo:
```powershell
.\.venv\Scripts\python.exe app/app.py
```
Navigate to: `http://127.0.0.1:7860`

---

## 6. Artifact & File Directory Mapping

| Purpose | Local Path | Description |
|---|---|---|
| **Trained Adapters** | `models/adapters/bioevidence-lora-best/` | Best LoRA adapter weights (`adapter_model.safetensors`). |
| **Experiment Metrics** | `outputs/experiments/mlflow.db` | Local SQLite database with step loss curves and parameters. |
| **Leakage Audit** | `outputs/data/leakage_report.json` | Verification showing zero train/eval article overlap. |
| **Quality Audit** | `outputs/evaluation/pilot_rubric_audit.json` | 100-sample human review rubric audit report. |
| **Unit Test Suite** | `tests/` | 16 automated tests (`pytest tests/`). |
