"""Supervised Fine-Tuning (SFT) Pipeline for BioEvidence-LLM.

Trains lightweight instruction models (e.g. Qwen2.5-1.5B/0.5B) using
4-bit QLoRA with BitsAndBytes and PEFT, logging experiments to MLflow.
Optimized for consumer GPUs with 4GB VRAM.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict

import mlflow
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from src.utils.config import get_model_config, get_training_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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
    quant_cfg = get_model_config()["quantization"]
    lora_cfg = get_model_config()["lora"]

    model_name = override_model_name or model_cfg["base_model_name_or_path"]
    output_dir = Path(train_cfg["output_dir"])
    adapter_dir = Path(train_cfg["adapter_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)

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
        logger.warning("MLflow initialization notice: %s", e)
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
    model.print_trainable_parameters()

    # Load SFT dataset
    logger.info("Loading SFT dataset from %s", dataset_file)
    dataset = load_dataset("json", data_files=dataset_file, split="train")

    if smoke_test:
        logger.info("Running SMOKE TEST (16 samples, 2 steps)")
        dataset = dataset.select(range(min(16, len(dataset))))
        max_steps = 2
        epochs = 1
    else:
        max_steps = -1
        epochs = train_cfg["num_train_epochs"]

    from trl import SFTConfig

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

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=sft_config,
    )

    logger.info("Starting SFT training pass...")
    train_result = trainer.train()

    # Save adapter
    logger.info("Saving best LoRA adapter to %s", adapter_dir)
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # Log metrics
    metrics = train_result.metrics
    logger.info("Training complete. Metrics: %s", metrics)
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
