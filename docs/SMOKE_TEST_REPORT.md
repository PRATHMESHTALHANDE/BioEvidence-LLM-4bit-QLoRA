# BioEvidence-LLM: Fine-Tuning Execution Report

> **Execution Date:** 2026-10-08 12:48:50 UTC  
> **Status:** COMPLETED  
> **Target Framework:** PyTorch CUDA | Transformers | PEFT QLoRA | TRL SFTTrainer  

## 1. Hardware & Model Architecture

| Parameter | Value |
|---|---|
| **Compute Device** | NVIDIA GeForce RTX 3050 Laptop GPU (Active Thermal) |
| **Memory Available** | 4.0 GB GDDR6 VRAM |
| **Base Model** | `Qwen/Qwen2.5-1.5B-Instruct` |
| **Quantization / Precision** | 4-bit NormalFloat4 (Double Quantization) |
| **PEFT Method** | LoRA ($r=16, \alpha=32$) |
| **Trainable Parameters** | **18,464,768** (1.182%) |
| **Dataset File** | `data/sft/BioEvidence-SFT-v0.1.jsonl` |
| **Learning Rate** | 0.0002 (Cosine Schedule) |
| **Effective Batch Size** | 1 $\times$ 8 = 8 |
| **Optimizer** | `paged_adamw_8bit` |
| **Gradient Checkpointing** | Enabled |

## 2. Step-by-Step Training Metrics

| Step | Training Loss | Learning Rate | Gradient Norm | Epoch |
|---|---|---|---|---|
| `1` | **2.2718** | `0.00e+00` | `4.41` | `0.50` |
| `2` | **2.0098** | `1.00e-04` | `3.39` | `1.00` |

## 3. Output Checkpoints & Artifact Locations

- **Saved LoRA Adapter:** [`models\adapters\bioevidence-lora-best`](file:///C:/Users/roari/Downloads/AI Projects/Fine Tunning LLM/models/adapters/bioevidence-lora-best)
- **Weights File:** `adapter_model.safetensors` (~36.9 MB)
- **MLflow Experiment DB:** `outputs/experiments/mlflow.db`
- **Gradio Web Demo:** [`app/app.py`](file:///c:/Users/roari/Downloads/AI%20Projects/Fine%20Tunning%20LLM/app/app.py)

## 4. Next Steps
1. Launch the **MLflow Dashboard** to view interactive step graphs: `.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///outputs/experiments/mlflow.db`
2. Launch the **Gradio Clinical Web Demo**: `.\.venv\Scripts\python.exe app/app.py`
3. Inspect the **Hugging Face Model Card**: [`models/adapters/README.md`](file:///c:/Users/roari/Downloads/AI%20Projects/Fine%20Tunning%20LLM/models/adapters/README.md)