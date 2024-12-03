import numpy as np 
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass
from datasets import load_dataset
from gluonts.dataset.arrow import ArrowWriter
from sklearn.gaussian_process import GaussianProcessRegressor
from tqdm.auto import tqdm
from sklearn.gaussian_process.kernels import (
    RBF, ConstantKernel, DotProduct, ExpSineSquared, 
    Kernel, RationalQuadratic, WhiteKernel
)
import functools

# Core datasets we'll use
CHRONOS_DATASETS = [
    "electricity_hourly",  # Common energy dataset
    "m4_hourly",          # M4 competition hourly data
    "traffic",            # Traffic volume data
    "weather",            # Weather time series
    "solar_5min"          # Solar power generation
]
KERNEL_BANK = [
    # Seasonal/Periodic patterns for different frequencies
    ExpSineSquared(periodicity=24),  # Daily
    ExpSineSquared(periodicity=168),  # Weekly
    ExpSineSquared(periodicity=730),  # Monthly
    ExpSineSquared(periodicity=8760),  # Yearly
    # Trend components
    DotProduct(sigma_0=0.0),  # Linear trend
    RBF(length_scale=0.1),  # Short-range correlations
    RBF(length_scale=1.0),  # Medium-range correlations
    RBF(length_scale=10.0),  # Long-range correlations
    # Other patterns
    RationalQuadratic(alpha=0.1),
    RationalQuadratic(alpha=1.0),
    WhiteKernel(noise_level=0.1),  # Noise component
    ConstantKernel(),  # Constant level
]


@dataclass
class TSMixupConfig:
    """Configuration for TSMixup augmentation"""
    max_series_to_mix: int = 3
    alpha: float = 1.5
    min_length: int = 128
    max_length: int = 512  # Reduced for testing
    scale_range: tuple = (-15.0, 15.0)

class TSMixup:
    def __init__(self, config: TSMixupConfig):
        self.config = config
        
    def mean_scale(self, series: np.ndarray) -> np.ndarray:
        scale = np.mean(np.abs(series))
        if scale == 0:
            scale = 1.0
        return series / scale
    
    def generate_single_mix(self, series_list: List[np.ndarray]) -> Dict:
        if not series_list:
            raise ValueError("series_list cannot be empty")
        
        # Number of series to mix (between 1 and max_series_to_mix)
        k = min(np.random.randint(1, self.config.max_series_to_mix + 1), len(series_list))
        
        # Select length
        length = np.random.randint(self.config.min_length, self.config.max_length + 1)
        
        # Select and process series
        selected_series = []
        indices = np.random.choice(len(series_list), k, replace=False)
        
        for idx in indices:
            series = series_list[idx]
            if len(series) > length:
                start_idx = np.random.randint(0, len(series) - length)
                series = series[start_idx:start_idx + length]
            else:
                pad_length = length - len(series)
                series = np.pad(series, (0, pad_length), mode='constant')
            selected_series.append(self.mean_scale(series))
        
        # Mix series using Dirichlet weights
        weights = np.random.dirichlet([self.config.alpha] * k)
        mixed_series = np.zeros(length)
        for w, s in zip(weights, selected_series):
            mixed_series += w * s
            
        return {
            "start": pd.Timestamp("2020-01-01"),
            "target": mixed_series.astype(np.float32)
        }

def load_chronos_datasets():
    """Load a subset of datasets from autogluon/chronos_datasets"""
    all_series = []
    
    for dataset_name in tqdm(CHRONOS_DATASETS, desc="Loading datasets"):
        try:
            print(f"\nLoading dataset: {dataset_name}")
            ds = load_dataset("autogluon/chronos_datasets", dataset_name)
            if 'train' in ds:
                series = ds['train']['target']
                # Take first 1000 series from each dataset for testing
                series = series[:1000]
                print(f"Loaded {len(series)} series from {dataset_name}")
                all_series.extend([np.array(s) for s in series])
        except Exception as e:
            print(f"Warning: Could not load dataset {dataset_name}: {str(e)}")
            continue
    
    print(f"\nTotal series loaded: {len(all_series)}")
    return all_series

def random_binary_map(a: Kernel, b: Kernel) -> Kernel:
    """Randomly combines two kernels using either addition or multiplication."""
    return np.random.choice([lambda x, y: x + y, lambda x, y: x * y])(a, b)

def generate_kernel_synth_ts(length: int = 512, max_kernels: int = 5) -> Dict:
    """
    Generate a synthetic time series using KernelSynth approach.
    
    Args:
        length: Length of the time series to generate
        max_kernels: Maximum number of kernels to combine
    """
    X = np.linspace(0, 1, length).reshape(-1, 1)
    
    # Randomly select and combine kernels
    n_kernels = np.random.randint(1, max_kernels + 1)
    selected_kernels = np.random.choice(KERNEL_BANK, n_kernels, replace=True)
    kernel = functools.reduce(random_binary_map, selected_kernels)
    
    # Sample from the GP prior
    try:
        gpr = GaussianProcessRegressor(kernel=kernel)
        ts = gpr.sample_y(X, n_samples=1, random_state=None).squeeze()
        return {
            "start": pd.Timestamp("2020-01-01"),
            "target": ts.astype(np.float32)
        }
    except np.linalg.LinAlgError:
        # Handle numerical stability issues by retrying
        return generate_kernel_synth_ts(length, max_kernels)

def generate_datasets(
    output_dir: str,
    n_synthetic: int = 1000,
    n_mixup: int = 1000,
    seed: Optional[int] = None
) -> None:
    """Generate both KernelSynth and TSMixup augmented datasets"""
    if seed is not None:
        np.random.seed(seed)
        
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data from HuggingFace
    print("Loading source datasets for TSMixup...")
    series_list = load_chronos_datasets()
    
    # Generate KernelSynth data
    print(f"\nGenerating {n_synthetic} KernelSynth time series...")
    synthetic_data = [
        generate_kernel_synth_ts()
        for _ in tqdm(range(n_synthetic), desc="Generating KernelSynth data")
    ]
    
    # Generate TSMixup augmentations
    print(f"\nGenerating {n_mixup} TSMixup augmentations...")
    mixup = TSMixup(TSMixupConfig())
    mixup_data = [
        mixup.generate_single_mix(series_list)
        for _ in tqdm(range(n_mixup), desc="Generating TSMixup data")
    ]
    
    # Save to Arrow files
    print("\nSaving datasets...")
    ArrowWriter(compression="lz4").write_to_file(
        synthetic_data,
        output_dir / "kernelsynth_data.arrow"
    )
    ArrowWriter(compression="lz4").write_to_file(
        mixup_data, 
        output_dir / "tsmixup_data.arrow"
    )