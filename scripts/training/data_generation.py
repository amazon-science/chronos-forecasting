import logging
import os
import sys
import json
import torch
import torch.distributed as dist
import numpy as np
from pathlib import Path
from typing import List, Dict, Iterator, Optional
from gluonts.dataset.common import FileDataset
from gluonts.transform import Filter
from functools import partial
from torch.utils.data import IterableDataset

from chronos.chronos import ChronosConfig, ChronosTokenizer
from chronos.base import BaseChronosPipeline
from chronos.utils import left_pad_and_stack_1D

# Reuse utility functions from original codebase
def is_main_process() -> bool:
    if not dist.is_torchelastic_launched():
        return True
    return int(os.environ["RANK"]) == 0

def log_on_main(msg: str, logger: logging.Logger, log_level: int = logging.INFO):
    if is_main_process():
        logger.log(log_level, msg)

class PseudoShuffledIterableDataset(IterableDataset):
    def __init__(self, base_dataset, shuffle_buffer_length: int = 100) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.shuffle_buffer_length = shuffle_buffer_length
        self.generator = torch.Generator()

    def __iter__(self):
        shuffle_buffer = []

        for element in self.base_dataset:
            shuffle_buffer.append(element)
            if len(shuffle_buffer) >= self.shuffle_buffer_length:
                idx = torch.randint(
                    len(shuffle_buffer), size=(), generator=self.generator
                )
                yield shuffle_buffer.pop(idx)

        while shuffle_buffer:
            idx = torch.randint(len(shuffle_buffer), size=(), generator=self.generator)
            yield shuffle_buffer.pop(idx)

class ShuffleMixin:
    def shuffle(self, shuffle_buffer_length: int = 100):
        return PseudoShuffledIterableDataset(self, shuffle_buffer_length)

class TSMixupDataset(IterableDataset, ShuffleMixin):
    def __init__(
        self,
        datasets: List[FileDataset],
        probabilities: List[float],
        k: int = 3,
        alpha: float = 1.5,
        l_min: int = 128,
        l_max: int = 2048,
        tokenizer: Optional[ChronosTokenizer] = None
    ) -> None:
        super().__init__()
        self.datasets = datasets
        self.probabilities = probabilities
        self.k = k
        self.alpha = alpha
        self.l_min = l_min
        self.l_max = l_max
        self.tokenizer = tokenizer

    def __iter__(self) -> Iterator:
        while True:
            # Sample k time series
            selected_series = []
            for _ in range(self.k):
                dataset_idx = np.random.choice(len(self.datasets), p=self.probabilities)
                series = np.random.choice(self.datasets[dataset_idx])
                selected_series.append(series)
                
            # Sample length
            l = np.random.randint(self.l_min, self.l_max)
            
            # Sample mixing weights
            weights = np.random.dirichlet([self.alpha] * self.k)
            
            # Create mixed series
            mixed_target = np.zeros(l)
            for w, series in zip(weights, selected_series):
                target = series["target"]
                scale = np.mean(np.abs(target))
                if scale == 0:
                    scale = 1.0
                scaled_target = target / scale
                
                if len(scaled_target) > l:
                    start_idx = np.random.randint(0, len(scaled_target) - l)
                    window = scaled_target[start_idx:start_idx + l]
                else:
                    padding = np.zeros(l - len(scaled_target))
                    window = np.concatenate([padding, scaled_target])
                
                mixed_target += w * window
            
            yield {
                "start": selected_series[0]["start"],
                "target": mixed_target
            }

def save_batch(dataset: TSMixupDataset, output_path: Path, batch_idx: int, batch_size: int):
    """Save a batch of data from the dataset."""
    batch_data = []
    batch_iter = iter(dataset)
    
    for _ in range(batch_size):
        batch_data.append(next(batch_iter))
    
    batch_path = output_path / f"batch_{batch_idx}.json"
    with open(batch_path, "w") as f:
        json.dump(batch_data, f)

def generate_and_save_data(
    data_paths: List[str],
    probabilities: List[float],
    output_dir: Path,
    total_samples: int = 1_000_000,
    batch_size: int = 1000,
    min_length: int = 64,
    max_missing_prop: float = 0.9,
    shuffle_buffer_length: int = 100,
    chronos_config: Optional[ChronosConfig] = None,
    logger: logging.Logger = None,
):
    """Generate and save TSMixup augmentations in batches using ChronosDataset."""
    if logger is None:
        logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        logger = logging.getLogger(__file__)
        logger.setLevel(logging.INFO)
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save generation info
    generation_info = {
        "data_paths": data_paths,
        "probabilities": probabilities,
        "total_samples": total_samples,
        "batch_size": batch_size,
        "min_length": min_length,
        "max_missing_prop": max_missing_prop,
        "shuffle_buffer_length": shuffle_buffer_length
    }
    with open(output_dir / "generation_info.json", "w") as f:
        json.dump(generation_info, f, indent=4)
    
    log_on_main(f"Loading datasets from {data_paths}", logger)
    
    # Load and filter datasets
    datasets = [
        Filter(
            partial(
                ChronosDataset.has_enough_observations,
                min_length=min_length,
                max_missing_prop=max_missing_prop,
            ),
            FileDataset(path=Path(data_path), freq="h"),
        )
        for data_path in data_paths
    ]
    
    # Create TSMixup dataset
    tokenizer = chronos_config.create_tokenizer() if chronos_config else None
    tsmixup_dataset = TSMixupDataset(
        datasets=datasets,
        probabilities=probabilities,
        l_min=min_length,
        l_max=chronos_config.context_length if chronos_config else 2048,
        tokenizer=tokenizer
    ).shuffle(shuffle_buffer_length=shuffle_buffer_length)
    
    num_batches = (total_samples + batch_size - 1) // batch_size
    log_on_main(f"Generating {num_batches} batches of size {batch_size}", logger)
    
    for batch_idx in range(num_batches):
        save_batch(tsmixup_dataset, output_dir, batch_idx, batch_size)
        log_on_main(f"Saved batch {batch_idx + 1}/{num_batches}", logger)

    # Save data config for training
    data_config = {
        "num_batches": num_batches,
        "batch_format": "batch_{}.json"
    }
    with open(output_dir / "data_config.json", "w") as f:
        json.dump(data_config, f, indent=4)