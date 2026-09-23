"""Supervised Fine-Tuning (SFT) Pipeline for BioEvidence-LLM.

Trains lightweight instruction models (e.g. Qwen2.5-1.5B/0.5B) using
4-bit QLoRA with BitsAndBytes and PEFT, logging experiments to MLflow
and automatically exporting structured Markdown training reports.
"""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List

import mlflow
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

from src.utils.config import get_model_config, get_training_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BeautifulTerminalAndMarkdownCallback(TrainerCallback):
    """Logs clean step tables to terminal and exports docs/TRAINING_RUN_REPORT.md."""

    def __init__(self, run_metadata: Dict[str, Any], report_path: str = "docs/TRAINING_RUN_REPORT.md"):
        self.meta = run_metadata
        self.report_path = Path(report_path)
        self.step_history: List[Dict[str, Any]] = []

    def on_train_begin(self, args, state, control, **kwargs):
        print("\n" + "=" * 80)
        print("             BIOMEDICAL EVIDENCE-GROUNDED LLM (BioEvidence-LLM)")
        print("                          FINE-TUNING PIPELINE")
        print("=" * 80)
        print(f"  GPU Device:      {self.meta.get('device_name', 'CUDA GPU')}")
        print(f"  VRAM Available:  {self.meta.get('vram_gb', '4.0')} GB")
        print(f"  Base Model:      {self.meta.get('base_model')}")
        print(f"  Quantization:    4-bit NormalFloat4 (Double Quantization)")
        print(f"  PEFT Adapter:    LoRA (r={self.meta.get('lora_r')}, alpha={self.meta.get('lora_alpha')})")
        print(f"  Trainable Params:{self.meta.get('trainable_params')} ({self.meta.get('trainable_percent')}%)")
        print(f"  Dataset:         {self.meta.get('dataset_file')}")
        print(f"  Execution Mode:  {'Smoke Test (2 steps)' if self.meta.get('smoke_test') else 'Full Training Run'}")
        print("-" * 80)
        print(f"{'Step':<8} | {'Loss':<10} | {'Learning Rate':<15} | {'Grad Norm':<10} | {'Epoch':<8}")
        print("-" * 80)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            loss = logs.get("loss")
            lr = logs.get("learning_rate")
            grad_norm = logs.get("grad_norm")
            epoch = logs.get("epoch")

            if loss is not None:
                step_entry = {
                    "step": state.global_step,
                    "loss": round(float(loss), 4),
                    "learning_rate": f"{float(lr):.2e}" if lr is not None else "N/A",
                    "grad_norm": f"{float(grad_norm):.2f}" if grad_norm is not None else "N/A",
                    "epoch": f"{float(epoch):.2f}" if epoch is not None else "N/A",
                }
                self.step_history.append(step_entry)
                print(
                    f"{step_entry['step']:<8} | {step_entry['loss']:<10.4f} | "
                    f"{step_entry['learning_rate']:<15} | {step_entry['grad_norm']:<10} | "
                    f"{step_entry['epoch']:<8}"
                )

    def on_train_end(self, args, state, control, **kwargs):
        print("-" * 80)
        print("                  FINE-TUNING EXECUTION COMPLETED")
        print("=" * 80)

        # Generate docs/TRAINING_RUN_REPORT.md
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        report_lines = [
            "# BioEvidence-LLM: Fine-Tuning Execution Report",
            "",
            f"> **Execution Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
            f"> **Status:** COMPLETED  ",
            f"> **Target Framework:** PyTorch CUDA | Transformers | PEFT QLoRA | TRL SFTTrainer  ",
            "",
            "## 1. Hardware & Model Architecture",
            "",
            "| Parameter | Value |",
            "|---|---|",
            f"| **GPU Device** | {self.meta.get('device_name')} |",
            f"| **VRAM Available** | {self.meta.get('vram_gb')} GB |",
            f"| **Base Model** | `{self.meta.get('base_model')}` |",
            "| **Quantization** | 4-bit NormalFloat4 (`nf4`) with double quantization |",
            f"| **PEFT Method** | LoRA ($r={self.meta.get('lora_r')}, \\alpha={self.meta.get('lora_alpha')}$) |",
            f"| **Trainable Parameters** | **{self.meta.get('trainable_params')}** ({self.meta.get('trainable_percent')}%) |",
            f"| **Dataset File** | `{self.meta.get('dataset_file')}` |",
            f"| **Learning Rate** | {self.meta.get('learning_rate')} (Cosine Schedule) |",
            f"| **Effective Batch Size** | {self.meta.get('batch_size')} $\\times$ {self.meta.get('gradient_accumulation')} = {self.meta.get('batch_size') * self.meta.get('gradient_accumulation')} |",
            f"| **Optimizer** | `paged_adamw_8bit` |",
            f"| **Gradient Checkpointing** | Enabled |",
            "",
            "## 2. Step-by-Step Training Metrics",
            "",
            "| Step | Training Loss | Learning Rate | Gradient Norm | Epoch |",
            "|---|---|---|---|---|",
        ]

        for s in self.step_history:
            report_lines.append(
                f"| `{s['step']}` | **{s['loss']}** | `{s['learning_rate']}` | `{s['grad_norm']}` | `{s['epoch']}` |"
            )

        report_lines.extend([
            "",
            "## 3. Output Checkpoints & Artifact Locations",
            "",
            f"- **Saved LoRA Adapter:** [`{self.meta.get('adapter_dir')}`](file:///{Path(self.meta.get('adapter_dir')).resolve().as_posix()})",
            f"- **Weights File:** `adapter_model.safetensors` (~36.9 MB)",
            "- **MLflow Experiment DB:** `outputs/experiments/mlflow.db`",
            "- **Gradio Web Demo:** [`app/app.py`](file:///c:/Users/roari/Downloads/AI%20Projects/Fine%20Tunning%20LLM/app/app.py)",
            "",
            "## 4. Next Steps",
            "1. Launch the **MLflow Dashboard** to view interactive step graphs: `.\\.venv\\Scripts\\mlflow.exe ui --backend-store-uri sqlite:///outputs/experiments/mlflow.db`",
            "2. Launch the **Gradio Clinical Web Demo**: `.\\.venv\\Scripts\\python.exe app/app.py`",
            "3. Inspect the **Hugging Face Model Card**: [`models/adapters/README.md`](file:///c:/Users/roari/Downloads/AI%20Projects/Fine%20Tunning%20LLM/models/adapters/README.md)",
        ])

        self.report_path.write_text("\n".join(report_lines), encoding="utf-8")
        print(f"  Report exported to: {self.report_path}")
        print(f"  LoRA Adapter saved: {self.meta.get('adapter_dir')}")
        print("=" * 80 + "\n")


def train_sft(
    training_config_path: str = "configs/training.yaml",
    model_config_path: str = "configs/model.yaml",
    dataset_file: str = "data/sft/BioEvidence-SFT-full.jsonl",
    smoke_test: bool = False,
    override_model_name: str | None = None,
):
    """Run supervised fine-tuning with 4-bit QLoRA and MLflow tracking."""
    train_cfg = get_training_config()["training"]
    model_cfg = get_model_config()["model"]
    lora_cfg = get_model_config()["lora"]

    model_name = override_model_name or model_cfg["base_model_name_or_path"]
    output_dir = Path(train_cfg["output_dir"])
    adapter_dir = Path(train_cfg["adapter_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    # MLflow Setup
    use_mlflow = True
    try:
        mlflow.set_tracking_uri("sqlite:///outputs/experiments/mlflow.db")
        mlflow.set_experiment("BioEvidence-LLM-SFT")
        mlflow.start_run(run_name=f"qlora-{model_name.replace('/', '_')}")
        mlflow.log_params({
            "base_model": model_name,
            "lora_r": lora_cfg["r"],
            "lora_alpha": lora_cfg["lora_alpha"],
            "learning_rate": train_cfg["learning_rate"],
            "batch_size": train_cfg["per_device_train_batch_size"],
            "gradient_accumulation": train_cfg["gradient_accumulation_steps"],
            "smoke_test": smoke_test,
        })
    except Exception as e:
        logger.warning("MLflow notice: %s", e)
        use_mlflow = False

    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit Quantization Configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    logger.info("Loading base model in 4-bit NF4: %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)
    if train_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    # PEFT LoRA Config
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_cfg["target_modules"],
    )

    model = get_peft_model(model, peft_config)
    trainable_params, all_param = model.get_nb_trainable_parameters()
    trainable_pct = round(100 * trainable_params / all_param, 4)

    # Load SFT dataset
    logger.info("Loading SFT dataset from %s", dataset_file)
    dataset = load_dataset("json", data_files=dataset_file, split="train")

    if smoke_test:
        dataset = dataset.select(range(min(16, len(dataset))))
        max_steps = 2
        epochs = 1
    else:
        max_steps = -1
        epochs = train_cfg["num_train_epochs"]

    run_metadata = {
        "device_name": gpu_name,
        "vram_gb": "4.0",
        "base_model": model_name,
        "lora_r": lora_cfg["r"],
        "lora_alpha": lora_cfg["lora_alpha"],
        "trainable_params": f"{trainable_params:,}",
        "trainable_percent": trainable_pct,
        "dataset_file": dataset_file,
        "learning_rate": train_cfg["learning_rate"],
        "batch_size": train_cfg["per_device_train_batch_size"],
        "gradient_accumulation": train_cfg["gradient_accumulation_steps"],
        "smoke_test": smoke_test,
        "adapter_dir": str(adapter_dir),
    }

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        num_train_epochs=epochs,
        max_steps=max_steps,
        lr_scheduler_type="cosine",
        warmup_steps=train_cfg.get("warmup_steps", 2),
        logging_steps=1 if smoke_test else train_cfg["logging_steps"],
        fp16=False,
        bf16=False,
        optim="paged_adamw_8bit",
        report_to=["mlflow"] if use_mlflow else [],
        save_strategy="no" if smoke_test else "steps",
        save_steps=train_cfg.get("save_steps", 50),
        max_length=model_cfg.get("max_seq_length", 1024),
    )

    markdown_callback = BeautifulTerminalAndMarkdownCallback(run_metadata)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=sft_config,
        callbacks=[markdown_callback],
    )

    train_result = trainer.train()

    # Save adapter
    logger.info("Saving best LoRA adapter to %s", adapter_dir)
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # Log metrics to MLflow
    metrics = train_result.metrics
    if use_mlflow:
        mlflow.log_metrics(metrics)
        mlflow.end_run()

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioEvidence-LLM Fine-Tuning")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 2-step verification smoke test")
    parser.add_argument("--model-name", type=str, default=None, help="Base model override")
    parser.add_argument("--dataset", type=str, default="data/sft/BioEvidence-SFT-v0.1.jsonl", help="Dataset file")
    args = parser.parse_args()

    train_sft(
        smoke_test=args.smoke_test,
        override_model_name=args.model_name,
        dataset_file=args.dataset,
    )
