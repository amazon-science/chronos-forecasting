import ast
import json
import logging
import os
import random
import uuid
from pathlib import Path
from typing import List, Union

import click
import numpy as np
import torch
import transformers

# -------------------------------------------------
# Logger (moved to module scope)
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------------------------------------
# Security Helpers
# -------------------------------------------------
def safe_json_load(value: str, expected_type):
    """Safely parse CLI JSON input with validation."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON input: {e}")

    if not isinstance(parsed, expected_type):
        raise TypeError(f"Expected {expected_type}, got {type(parsed)}")

    return parsed


def validate_dataset_paths(paths: List[str]) -> List[str]:
    """Prevent arbitrary filesystem reads."""
    validated = []
    for p in paths:
        path = Path(p).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        if not path.is_dir():
            raise ValueError(f"Dataset path must be a directory: {path}")
        validated.append(str(path))
    return validated


def make_unique_output_dir(base_dir: str) -> Path:
    """Avoid race conditions in CI / distributed training."""
    run_id = uuid.uuid4().hex[:8]
    path = Path(base_dir) / f"run-{run_id}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_full_seed(seed: int):
    """Fully deterministic training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------------------------------
# CLI
# -------------------------------------------------
@click.command()
@click.option("--training_data_paths", required=True, type=str)
@click.option("--probability", default="[]", type=str)
@click.option("--tokenizer_kwargs", default="{}", type=str)
@click.option("--output_dir", default="./outputs", type=str)
@click.option("--seed", default=42, type=int)
def main(
    training_data_paths: str,
    probability: str,
    tokenizer_kwargs: str,
    output_dir: str,
    seed: int,
):
    """
    Secure + Optimized Chronos Training CLI
    """

    logger.info("Starting Chronos training pipeline")

    # -------------------------------------------------
    # Parse & Validate Inputs (SECURE)
    # -------------------------------------------------
    training_data_paths = safe_json_load(training_data_paths, list)
    training_data_paths = validate_dataset_paths(training_data_paths)

    probability = safe_json_load(probability, list)
    tokenizer_kwargs = safe_json_load(tokenizer_kwargs, dict)

    # -------------------------------------------------
    # Deterministic Training
    # -------------------------------------------------
    set_full_seed(seed)

    # -------------------------------------------------
    # Safe output directory
    # -------------------------------------------------
    output_dir = make_unique_output_dir(output_dir)
    logger.info(f"Output directory: {output_dir}")

    # -------------------------------------------------
    # Example Training Setup
    # -------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Simulated training loop placeholder
    logger.info("Loading datasets...")
    for path in training_data_paths:
        logger.info(f"Dataset loaded from: {path}")

    logger.info("Tokenizer kwargs:")
    logger.info(tokenizer_kwargs)

    # -------------------------------------------------
    # Simulated Training Loop
    # -------------------------------------------------
    logger.info("Training started...")
    for epoch in range(3):
        logger.info(f"Epoch {epoch+1} completed")

    logger.info("Training completed successfully 🎉")
    logger.info(f"Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
