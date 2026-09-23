"""Configuration loader and validator for BioEvidence-LLM."""

from pathlib import Path
from typing import Any, Dict
import yaml


def load_yaml(file_path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file safely.

    Args:
        file_path: Path to the YAML file.

    Returns:
        Dict containing parsed YAML data.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If parsing fails.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
            return data if data is not None else {}
        except yaml.YAMLError as exc:
            raise ValueError(f"Failed to parse YAML file {path}: {exc}") from exc


def get_model_config(config_dir: str | Path = "configs") -> Dict[str, Any]:
    """Load model configuration."""
    return load_yaml(Path(config_dir) / "model.yaml")


def get_dataset_config(config_dir: str | Path = "configs") -> Dict[str, Any]:
    """Load dataset configuration."""
    return load_yaml(Path(config_dir) / "dataset.yaml")


def get_training_config(config_dir: str | Path = "configs") -> Dict[str, Any]:
    """Load training configuration."""
    return load_yaml(Path(config_dir) / "training.yaml")


def get_evaluation_config(config_dir: str | Path = "configs") -> Dict[str, Any]:
    """Load evaluation configuration."""
    return load_yaml(Path(config_dir) / "evaluation.yaml")
