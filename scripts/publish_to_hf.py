"""Hugging Face Hub Publishing Script for BioEvidence-LLM.

Publishes:
1. Fine-tuned LoRA Adapter weights to: https://huggingface.co/Bhupati1998/BioEvidence-LLM-1.5B
2. SFT and Evaluation Datasets to: https://huggingface.co/datasets/Bhupati1998/BioEvidence-SFT

Reads credentials from .env or environment variable HF_TOKEN.
"""

import argparse
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load .env if present
load_dotenv()


def publish_model(
    repo_id: str = "Bhupati1998/BioEvidence-LLM-1.5B",
    adapter_dir: str = "models/adapters/bioevidence-lora-best",
    token: str | None = None,
    private: bool = False,
):
    """Upload trained LoRA adapter and model card to Hugging Face Hub."""
    hf_token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        try:
            from huggingface_hub import get_token
            hf_token = get_token()
        except ImportError:
            pass

    if not hf_token:
        raise ValueError(
            "Hugging Face write token not found! Please provide it via:\n"
            "  1. .env file: HF_TOKEN=hf_...\n"
            "  2. CLI flag: --token hf_...\n"
            "  3. Or run: .\\.venv\\Scripts\\huggingface-cli.exe login\n"
            "Get your write token from: https://huggingface.co/settings/tokens"
        )

    api = HfApi(token=hf_token)
    adapter_path = Path(adapter_dir)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter folder not found at: {adapter_path.resolve()}")

    logger.info("Verifying/creating Hugging Face model repository: %s", repo_id)
    create_repo(repo_id=repo_id, token=hf_token, private=private, exist_ok=True)

    logger.info("Uploading adapter weights and artifacts from %s...", adapter_path)
    api.upload_folder(
        folder_path=str(adapter_path),
        repo_id=repo_id,
        repo_type="model",
        token=hf_token,
    )

    url = f"https://huggingface.co/{repo_id}"
    logger.info("Successfully published model to: %s", url)
    print("\n" + "=" * 80)
    print(f"🎉 Model adapter successfully published to Hugging Face:")
    print(f"   {url}")
    print("=" * 80 + "\n")


def publish_dataset(
    repo_id: str = "Bhupati1998/BioEvidence-Datasets",
    sft_file: str = "data/sft/BioEvidence-SFT-full.jsonl",
    eval_file: str = "data/evaluation/BioEvidence-Eval-v0.1.jsonl",
    token: str | None = None,
    private: bool = False,
):
    """Upload SFT and Evaluation datasets to Hugging Face Hub."""
    hf_token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        try:
            from huggingface_hub import get_token
            hf_token = get_token()
        except ImportError:
            pass

    if not hf_token:
        raise ValueError(
            "Hugging Face write token not found! Please provide it via:\n"
            "  1. .env file: HF_TOKEN=hf_...\n"
            "  2. CLI flag: --token hf_...\n"
            "  3. Or run: .\\.venv\\Scripts\\huggingface-cli.exe login\n"
            "Get your write token from: https://huggingface.co/settings/tokens"
        )

    api = HfApi(token=hf_token)
    logger.info("Verifying/creating Hugging Face dataset repository: %s", repo_id)
    create_repo(repo_id=repo_id, repo_type="dataset", token=hf_token, private=private, exist_ok=True)

    sft_path = Path(sft_file)
    eval_path = Path(eval_file)
    card_path = Path("data/sft/README.md")

    if sft_path.exists():
        logger.info("Uploading training dataset: %s", sft_path)
        api.upload_file(
            path_or_fileobj=str(sft_path),
            path_in_repo="train.jsonl",
            repo_id=repo_id,
            repo_type="dataset",
            token=hf_token,
        )

    if eval_path.exists():
        logger.info("Uploading held-out evaluation dataset: %s", eval_path)
        api.upload_file(
            path_or_fileobj=str(eval_path),
            path_in_repo="eval.jsonl",
            repo_id=repo_id,
            repo_type="dataset",
            token=hf_token,
        )

    if card_path.exists():
        logger.info("Uploading Dataset Card: %s", card_path)
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=hf_token,
        )

    url = f"https://huggingface.co/datasets/{repo_id}"
    logger.info("Successfully published dataset to: %s", url)
    print("\n" + "=" * 80)
    print(f"🎉 Datasets successfully published to Hugging Face:")
    print(f"   {url}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish BioEvidence-LLM to Hugging Face Hub")
    parser.add_argument("--model", action="store_true", help="Publish trained model adapter")
    parser.add_argument("--dataset", action="store_true", help="Publish SFT and evaluation datasets")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face Access Token (WRITE)")
    parser.add_argument("--model-repo", type=str, default="Bhupati1998/BioEvidence-LLM-1.5B", help="Model Repo ID")
    parser.add_argument("--dataset-repo", type=str, default="Bhupati1998/BioEvidence-Datasets", help="Dataset Repo ID")
    parser.add_argument("--private", action="store_true", help="Create as private repository")
    args = parser.parse_args()

    if not args.model and not args.dataset:
        # Default to publishing model if none specified
        args.model = True

    if args.model:
        publish_model(repo_id=args.model_repo, token=args.token, private=args.private)

    if args.dataset:
        publish_dataset(repo_id=args.dataset_repo, token=args.token, private=args.private)
