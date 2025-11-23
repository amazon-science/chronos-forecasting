# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Authors: Abdul Fatir Ansari <ansarnd@amazon.com>

import warnings
from typing import TYPE_CHECKING, cast

from torch.utils.data import DataLoader, Dataset
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback

if TYPE_CHECKING:
    from chronos.chronos2.dataset import Chronos2Dataset


def seed_worker(worker_id: int):
    import random

    import numpy as np
    import torch

    seed = torch.initial_seed() % 2**32 + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EvaluateAndSaveFinalStepCallback(TrainerCallback):
    """Callback to evaluate and save the model at last training step."""

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= state.max_steps:
            control.should_log = True
            control.should_evaluate = True
            control.should_save = True


class Chronos2Trainer(Trainer):
    """
    A custom trainer based on transformers Trainer. We need to override the dataloader getters because we handle
    batching ourselves in a custom dataset which directly returns batches instead of individual elements.
    """

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = cast("Chronos2Dataset", self.train_dataset)

        if self.args.train_batch_size > train_dataset.batch_size:
            warnings.warn(
                f"The batch_size of the train_dataset ({train_dataset.batch_size}) does not match the batch_size "
                f"in TrainingArguments ({self.args.train_batch_size}). On machines with multiple GPUs, this may indicate "
                f"that multiple GPUs are visible and transformers is using DataParallel for training by default. "
                f"This may lead to unnecessary slowdown and unexpected behavior. We strongly recommend setting the CUDA_VISIBLE_DEVICES "
                f"environment variable to ensure that only a single GPU is visible.",
                category=UserWarning,
                stacklevel=3,
            )

        dataloader_params = {
            # Disable automatic batching as we handle batching ourselves
            "batch_size": None,
            "collate_fn": None,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "drop_last": self.args.dataloader_drop_last,
            "worker_init_fn": seed_worker,
            "prefetch_factor": self.args.dataloader_prefetch_factor,
        }

        return DataLoader(train_dataset, **dataloader_params)  # type: ignore

    def get_eval_dataloader(self, eval_dataset: str | Dataset | None = None) -> DataLoader:
        if self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = cast("Chronos2Dataset", self.eval_dataset)

        if self.args.eval_batch_size > eval_dataset.batch_size:
            warnings.warn(
                f"The batch_size of the eval_dataset ({eval_dataset.batch_size}) does not match the batch_size "
                f"in TrainingArguments ({self.args.eval_batch_size}). On machines with multiple GPUs, this may indicate "
                f"that multiple GPUs are visible and transformers is using DataParallel for training by default. "
                f"This may lead to unnecessary slowdown and unexpected behavior. We strongly recommend setting the CUDA_VISIBLE_DEVICES "
                f"environment variable to ensure that only a single GPU is visible.",
                category=UserWarning,
                stacklevel=3,
            )

        dataloader_params = {
            # Disable automatic batching as we handle batching ourselves
            "batch_size": None,
            "collate_fn": None,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "drop_last": self.args.dataloader_drop_last,
            "prefetch_factor": self.args.dataloader_prefetch_factor,
        }

        return DataLoader(eval_dataset, **dataloader_params)  # type: ignore
