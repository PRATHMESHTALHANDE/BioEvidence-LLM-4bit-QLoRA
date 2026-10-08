# BioEvidence-LLM: Fine-Tuning Execution Report

> **Execution Date:** 2026-10-08 15:04:24 UTC  
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
| **Dataset File** | `data/sft/BioEvidence-SFT-full.jsonl` |
| **Learning Rate** | 0.0002 (Cosine Schedule) |
| **Effective Batch Size** | 1 $\times$ 8 = 8 |
| **Optimizer** | `paged_adamw_8bit` |
| **Gradient Checkpointing** | Enabled |

## 2. Step-by-Step Training Metrics

| Step | Training Loss | Learning Rate | Gradient Norm | Epoch |
|---|---|---|---|---|
| `10` | **1.6904** | `2.00e-04` | `0.64` | `0.09` |
| `20` | **0.8812** | `1.99e-04` | `0.23` | `0.18` |
| `30` | **0.8131** | `1.97e-04` | `0.19` | `0.27` |
| `40` | **0.7878** | `1.94e-04` | `0.18` | `0.36` |
| `50` | **0.8035** | `1.90e-04` | `0.17` | `0.45` |
| `60` | **0.7631** | `1.85e-04` | `0.18` | `0.55` |
| `70` | **0.7964** | `1.80e-04` | `0.15` | `0.64` |
| `80` | **0.7826** | `1.74e-04` | `0.15` | `0.73` |
| `90` | **0.831** | `1.67e-04` | `0.15` | `0.82` |
| `100` | **0.7828** | `1.60e-04` | `0.17` | `0.91` |
| `110` | **0.7987** | `1.52e-04` | `0.15` | `1.00` |
| `120` | **0.7497** | `1.44e-04` | `0.16` | `1.09` |
| `130` | **0.7501** | `1.35e-04` | `0.17` | `1.18` |
| `140` | **0.7562** | `1.26e-04` | `0.16` | `1.27` |
| `150` | **0.7591** | `1.16e-04` | `0.20` | `1.36` |
| `160` | **0.7702** | `1.07e-04` | `0.20` | `1.45` |
| `170` | **0.7236** | `9.71e-05` | `0.19` | `1.55` |
| `180` | **0.7469** | `8.76e-05` | `0.21` | `1.64` |
| `190` | **0.7604** | `7.81e-05` | `0.19` | `1.73` |
| `200` | **0.7463** | `6.89e-05` | `0.19` | `1.82` |
| `210` | **0.7588** | `6.00e-05` | `0.21` | `1.91` |
| `220` | **0.7359** | `5.14e-05` | `0.20` | `2.00` |
| `230` | **0.7213** | `4.33e-05` | `0.20` | `2.09` |
| `240` | **0.7081** | `3.56e-05` | `0.23` | `2.18` |
| `250` | **0.702** | `2.86e-05` | `0.20` | `2.27` |
| `260` | **0.7082** | `2.22e-05` | `0.19` | `2.36` |
| `270` | **0.7417** | `1.66e-05` | `0.23` | `2.45` |
| `280` | **0.6824** | `1.17e-05` | `0.22` | `2.55` |
| `290` | **0.7153** | `7.61e-06` | `0.21` | `2.64` |
| `300` | **0.7336** | `4.38e-06` | `0.20` | `2.73` |
| `310` | **0.7044** | `2.02e-06` | `0.21` | `2.82` |
| `320` | **0.7212** | `5.55e-07` | `0.22` | `2.91` |
| `330` | **0.7098** | `4.59e-09` | `0.20` | `3.00` |

## 3. Output Checkpoints & Artifact Locations

- **Saved LoRA Adapter:** [`models\adapters\bioevidence-lora-best`](file:///C:/Users/roari/Downloads/AI Projects/Fine Tunning LLM/models/adapters/bioevidence-lora-best)
- **Weights File:** `adapter_model.safetensors` (~36.9 MB)
- **MLflow Experiment DB:** `outputs/experiments/mlflow.db`
- **Gradio Web Demo:** [`app/app.py`](file:///c:/Users/roari/Downloads/AI%20Projects/Fine%20Tunning%20LLM/app/app.py)

## 4. Next Steps
1. Launch the **MLflow Dashboard** to view interactive step graphs: `.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///outputs/experiments/mlflow.db`
2. Launch the **Gradio Clinical Web Demo**: `.\.venv\Scripts\python.exe app/app.py`
3. Inspect the **Hugging Face Model Card**: [`models/adapters/README.md`](file:///c:/Users/roari/Downloads/AI%20Projects/Fine%20Tunning%20LLM/models/adapters/README.md)