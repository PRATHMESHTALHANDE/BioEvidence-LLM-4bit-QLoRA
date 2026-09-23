"""Unit tests for configuration loading and validation."""

from pathlib import Path
from src.utils.config import (
    get_dataset_config,
    get_evaluation_config,
    get_model_config,
    get_training_config,
    load_yaml,
)


def test_load_model_config():
    """Verify model.yaml loads correctly with expected fields."""
    cfg = get_model_config()
    assert "model" in cfg
    assert "base_model_name_or_path" in cfg["model"]
    assert "quantization" in cfg
    assert cfg["quantization"]["load_in_4bit"] is True
    assert "lora" in cfg
    assert cfg["lora"]["r"] == 16


def test_load_dataset_config():
    """Verify dataset.yaml loads correctly."""
    cfg = get_dataset_config()
    assert "datasets" in cfg
    assert "pubmedqa" in cfg["datasets"]
    assert "splits" in cfg
    assert cfg["splits"]["group_by"] == "pmid"


def test_load_training_config():
    """Verify training.yaml loads correctly."""
    cfg = get_training_config()
    assert "training" in cfg
    assert cfg["training"]["learning_rate"] == 0.0002
    assert cfg["training"]["gradient_checkpointing"] is True


def test_load_evaluation_config():
    """Verify evaluation.yaml loads correctly."""
    cfg = get_evaluation_config()
    assert "evaluation" in cfg
    assert "metrics" in cfg
    assert "decision_accuracy" in cfg["metrics"]
