# Hardware & Environment Audit Report: BioEvidence-LLM

> **Date Generated:** 2026-09-23  
> **Status:** Phase 0 Completed — Ready for User Review  
> **Target Framework:** Python 3.11 | PyTorch (CUDA) | Hugging Face Transformers | TRL | PEFT | MLflow | Gradio

---

## 1. System Hardware Specifications

| Component | Specification / Detected Value | Status / Notes |
|---|---|---|
| **Operating System** | Microsoft Windows 11 Home Single Language (64-bit) | Supported |
| **CPU** | 12th Gen Intel(R) Core(TM) i7-12650H (10 Cores, 16 Threads) | High performance |
| **System RAM** | 16.0 GB (15.63 GB visible, ~4.8 GB free) | Sufficient for data prep & inference |
| **GPU** | NVIDIA GeForce RTX 3050 Laptop GPU | Active |
| **VRAM** | **4,096 MiB (4.0 GB GDDR6)** | **Critical constraint** for training |
| **NVIDIA Driver** | 561.17 | Up to date |
| **Max CUDA Supported** | CUDA 12.6 | Supports modern PyTorch CUDA 12.1/12.4 builds |
| **Storage (C:)** | ~761 GB free out of ~1 TB | Ample storage for datasets & weights |

---

## 2. Software & Tooling Audit

| Tool | Version / Path | Readiness |
|---|---|---|
| **Git** | `git version 2.55.0.windows.4` | Installed & verified |
| **Package Manager** | `uv 0.12.5` | Installed & verified |
| **Python Runtimes** | Python 3.11.16 (`C:\Users\roari\AppData\Roaming\uv\python\cpython-3.11.16-windows-x86_64-none\python.exe`)<br>Python 3.12.10 (`C:\Users\roari\AppData\Local\Programs\Python\Python312\python.exe`) | **Python 3.11.16 selected** (optimal for PyTorch + CUDA + ML libraries) |
| **PyTorch & CUDA** | Not installed in environment | To be installed cleanly in `.venv/` with CUDA 12.1/12.4 wheels |
| **Hugging Face Stack** | Not installed | To be installed in Phase 1 (`transformers`, `peft`, `trl`, `datasets`, `accelerate`) |
| **BitsAndBytes / Unsloth** | Not installed | `bitsandbytes` (Windows prebuilt wheels) for 4-bit QLoRA; Unsloth tested in Phase 1 |

---

## 3. Feasible Model Size Range & Quantization Strategy

Given the **4.0 GB VRAM** hardware profile:

### 3.1 Training Feasibility Assessment
* **7B / 8B Models (e.g., Llama-3-8B, Qwen2.5-7B):**
  * *Status:* **Infeasible for local fine-tuning on 4GB VRAM**.
  * *Reason:* 4-bit base weights take ~4.5 GB alone before allocating optimizer states, adapter weights, and sequence activations.
* **3B Models (e.g., Qwen2.5-3B, Llama-3.2-3B):**
  * *Status:* **Marginal / Experimental**. Requires aggressive 4-bit QLoRA, sequence length $\le 1024$, batch size 1, gradient checkpointing, and gradient accumulation.
* **1.5B / 0.5B Models (e.g., Qwen2.5-1.5B, Qwen2.5-0.5B, Llama-3.2-1B):**
  * *Status:* **Highly Feasible & Recommended (Sweet Spot)**.
  * 4-bit QLoRA base model occupies ~1.1 GB VRAM.
  * LoRA adapters + AdamW optimizer + activations fit comfortably within ~2.5–3.2 GB VRAM.
  * Allows sequence lengths up to 2048 tokens (essential for biomedical abstracts and multi-task JSON outputs).

### 3.2 Primary Recommended Base Model
* **Model:** `Qwen/Qwen2.5-1.5B-Instruct` (or `Qwen/Qwen2.5-0.5B-Instruct` for ultra-fast pilot iterations).
* **Quantization Strategy:** 4-bit QLoRA (`bitsandbytes` `nf4` with double quantization) + LoRA rank $r=16, \alpha=32$ on all linear projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
* **Sequence Length:** 1024–2048 tokens with gradient checkpointing enabled.

---

## 4. Phase 0 Recommendation & Next Steps

1. **Virtual Environment Setup (Phase 1):** Create a dedicated `.venv` using the verified Python 3.11.16 interpreter.
2. **CUDA PyTorch Wheel Installation:** Install PyTorch with CUDA 12.1/12.4 support into `.venv`.
3. **Core Dependencies:** Install `transformers`, `datasets`, `accelerate`, `peft`, `trl`, `bitsandbytes`, `mlflow`, `gradio`, `pydantic`.
4. **Git Repository Setup (Phase 2):** Run `git init` and establish strict `.gitignore`.
