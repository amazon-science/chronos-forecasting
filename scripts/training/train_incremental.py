import ast
import logging
import os
import re
import sys
import json
import itertools
import random
from copy import deepcopy
from pathlib import Path
from functools import partial
from typing import List, Iterator, Optional, Dict

from chronos.chronos import ChronosConfig
from scripts.training.train import ChronosDataset, get_next_path
import typer
from typer_config import use_yaml_config
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info
import transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoConfig,
    T5Config,
    Trainer,
    TrainingArguments,
)
import accelerate
import gluonts
from gluonts.dataset.common import FileDataset
from gluonts.itertools import Cyclic, Map, Filter
from gluonts.transform import (
    FilterTransformation,
    TestSplitSampler,
    ValidationSplitSampler,
    InstanceSplitter,
    ExpectedNumInstanceSampler,
    MissingValueImputation,
    LeavesMissingValues,
    LastValueImputation,
)

app = typer.Typer(pretty_exceptions_enable=False)

def is_main_process() -> bool:
    if not dist.is_torchelastic_launched():
        return True
    return int(os.environ["RANK"]) == 0

def log_on_main(msg: str, logger: logging.Logger, log_level: int = logging.INFO):
    if is_main_process():
        logger.log(log_level, msg)


def get_training_job_info() -> Dict:
    job_info = {}
    job_info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        job_info["device_count"] = torch.cuda.device_count()
        job_info["device_names"] = {
            idx: torch.cuda.get_device_name(idx)
            for idx in range(torch.cuda.device_count())
        }
        job_info["mem_info"] = {
            idx: torch.cuda.mem_get_info(device=idx)
            for idx in range(torch.cuda.device_count())
        }
    job_info["torchelastic_launched"] = dist.is_torchelastic_launched()
    if dist.is_torchelastic_launched():
        job_info["world_size"] = dist.get_world_size()
    return job_info

def save_training_info(ckpt_path: Path, training_config: Dict):
    assert ckpt_path.is_dir()
    with open(ckpt_path / "training_info.json", "w") as fp:
        json.dump(
            {"training_config": training_config, "job_info": get_training_job_info()},
            fp,
            indent=4,
        )

def load_model(
    model_id="google/t5-efficient-tiny",
    model_type="seq2seq",
    vocab_size=4096,
    random_init=False,
    tie_embeddings=False,
    pad_token_id=0,
    eos_token_id=1,
):
    """Reuse original load_model function."""
    assert model_type in ["seq2seq", "causal"]
    AutoModelClass = (
        AutoModelForSeq2SeqLM if model_type == "seq2seq" else AutoModelForCausalLM
    )
    if random_init:
        log_on_main("Using random initialization", logger)
        config = AutoConfig.from_pretrained(model_id)
        if isinstance(config, T5Config):
            config.initializer_factor = 0.05
        config.tie_word_embeddings = tie_embeddings
        model = AutoModelClass.from_config(config)
    else:
        log_on_main(f"Using pretrained initialization from {model_id}", logger)
        model = AutoModelClass.from_pretrained(model_id)

    model.resize_token_embeddings(vocab_size)
    model.config.pad_token_id = model.generation_config.pad_token_id = pad_token_id
    model.config.eos_token_id = model.generation_config.eos_token_id = eos_token_id
    return model

def load_batch_datasets(
    data_dir: Path,
    start_batch: int,
    end_batch: int,
    logger: logging.Logger
) -> List[FileDataset]:
    """Load a range of data batches."""
    datasets = []
    log_on_main(f"Loading batches {start_batch} to {end_batch}", logger)
    for batch_idx in range(start_batch, end_batch):
        batch_path = data_dir / f"batch_{batch_idx}.json"
        if batch_path.exists():
            datasets.append(FileDataset(path=batch_path, freq="h"))
    return datasets

def train_on_batches(
    data_dir: Path,
    output_dir: Path,
    model_path: Optional[str],
    start_batch: int,
    end_batch: int,
    training_args: dict,
    logger: logging.Logger
):
    """Train model on a range of data batches."""
    
    # Load or initialize model
    if model_path:
        log_on_main(f"Loading model from {model_path}", logger)
        model = load_model(model_path)
    else:
        log_on_main("Initializing new model", logger)
        model = load_model(**training_args["model_args"])
    
    # Load datasets
    datasets = load_batch_datasets(data_dir, start_batch, end_batch, logger)
    
    # Create training arguments
    train_args = TrainingArguments(
        output_dir=str(output_dir),
        **training_args["training_args"]
    )
    
    # Create trainer and train
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=datasets,
    )
    
    log_on_main("Starting training", logger)
    trainer.train()
    
    # Save checkpoint and training info
    checkpoint_dir = output_dir / f"checkpoint-batch-{end_batch}"
    model.save_pretrained(checkpoint_dir)
    save_training_info(checkpoint_dir, training_args)
    log_on_main(f"Saved checkpoint to {checkpoint_dir}", logger)

@app.command()
@use_yaml_config(param_name="config")
def main(
    # Data params
    data_dir: str,
    batch_start: int,
    batch_end: int,
    context_length: int = 512,
    prediction_length: int = 64,
    min_past: int = 64,
    max_missing_prop: float = 0.9,
    shuffle_buffer_length: int = 100,
    
    # Training params
    max_steps: int = 200_000,
    save_steps: int = 50_000,
    log_steps: int = 500,
    per_device_train_batch_size: int = 32,
    learning_rate: float = 1e-3,
    optim: str = "adamw_torch_fused",
    gradient_accumulation_steps: int = 2,
    
    # Model params
    model_id: str = "google/t5-efficient-tiny",
    model_type: str = "seq2seq",
    random_init: bool = False,
    tie_embeddings: bool = False,
    model_path: Optional[str] = None,
    
    # Output params
    output_dir: str = "./output/",
    
    # Hardware params
    tf32: bool = True,
    torch_compile: bool = True,
    dataloader_num_workers: int = 1,
    
    # Tokenizer params
    tokenizer_class: str = "MeanScaleUniformBins",
    tokenizer_kwargs: str = "{'low_limit': -15.0, 'high_limit': 15.0}",
    n_tokens: int = 4096,
    n_special_tokens: int = 2,
    pad_token_id: int = 0,
    eos_token_id: int = 1,
    use_eos_token: bool = True,
    
    # Additional training params
    lr_scheduler_type: str = "linear",
    warmup_ratio: float = 0.0,
    num_samples: int = 20,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    seed: Optional[int] = None,
):
    if tf32 and not (
        torch.cuda.is_available() and torch.cuda.get_device_capabilities()[0] >= 8
    ):
        log_on_main(
            "TF32 format is only available on devices with compute capability >= 8. "
            "Setting tf32 to False.",
            logger,
        )
        tf32 = False

    if seed is None:
        seed = random.randint(0, 2**32)

    log_on_main(f"Using SEED: {seed}", logger)
    transformers.set_seed(seed=seed)

    raw_training_config = deepcopy(locals())
    output_dir = Path(output_dir)
    data_dir = Path(data_dir)

    if isinstance(tokenizer_kwargs, str):
        tokenizer_kwargs = ast.literal_eval(tokenizer_kwargs)
    assert isinstance(tokenizer_kwargs, dict)

    assert model_type in ["seq2seq", "causal"]

    output_dir = get_next_path("run", base_dir=output_dir, file_type="")
    log_on_main(f"Logging dir: {output_dir}", logger)
    
    # Load data config
    with open(data_dir / "data_config.json", "r") as f:
        data_config = json.load(f)
    total_batches = data_config["num_batches"]
    
    if batch_end > total_batches:
        raise ValueError(f"batch_end {batch_end} exceeds total batches {total_batches}")
    
    log_on_main(
        f"Loading and processing batches {batch_start} to {batch_end} "
        f"out of {total_batches} total batches",
        logger,
    )

    train_datasets = [
        FileDataset(path=data_dir / f"batch_{i}.json", freq="h")
        for i in range(batch_start, batch_end)
    ]

    log_on_main("Initializing model", logger)
    
    if model_path:
        log_on_main(f"Loading model from {model_path}", logger)
        model = load_model(
            model_id=model_path,
            model_type=model_type,
            vocab_size=n_tokens,
            random_init=False,
            tie_embeddings=tie_embeddings,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
    else:
        model = load_model(
            model_id=model_id,
            model_type=model_type,
            vocab_size=n_tokens,
            random_init=random_init,
            tie_embeddings=tie_embeddings,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )

    chronos_config = ChronosConfig(
        tokenizer_class=tokenizer_class,
        tokenizer_kwargs=tokenizer_kwargs,
        n_tokens=n_tokens,
        n_special_tokens=n_special_tokens,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        use_eos_token=use_eos_token,
        model_type=model_type,
        context_length=context_length,
        prediction_length=prediction_length,
        num_samples=num_samples,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    # Add extra items to model config so that it's saved in the ckpt
    model.config.chronos_config = chronos_config.__dict__

    shuffled_train_dataset = ChronosDataset(
        datasets=train_datasets,
        probabilities=[1.0 / len(train_datasets)] * len(train_datasets),
        tokenizer=chronos_config.create_tokenizer(),
        context_length=context_length,
        prediction_length=prediction_length,
        min_past=min_past,
        model_type=model_type,
        imputation_method=LastValueImputation() if model_type == "causal" else None,
        mode="training",
    ).shuffle(shuffle_buffer_length=shuffle_buffer_length)

    # Define training args
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        optim=optim,
        logging_dir=str(output_dir / "logs"),
        logging_strategy="steps",
        logging_steps=log_steps,
        save_strategy="steps",
        save_steps=save_steps,
        report_to=["tensorboard"],
        max_steps=max_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_num_workers=dataloader_num_workers,
        tf32=tf32,
        torch_compile=torch_compile,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=shuffled_train_dataset,
    )
    
    log_on_main("Starting training", logger)
    trainer.train()

    if is_main_process():
        checkpoint_dir = output_dir / f"checkpoint-batch-{batch_end}"
        model.save_pretrained(checkpoint_dir)
        save_training_info(checkpoint_dir, training_config=raw_training_config)
        log_on_main(f"Saved checkpoint to {checkpoint_dir}", logger)

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)
    app()