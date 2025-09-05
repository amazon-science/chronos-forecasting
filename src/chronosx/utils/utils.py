from torch import nn
import logging
import numpy as np
import os
import torch
import torch.distributed as dist


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_metrics(pred, num_classes=4096):
    loss = nn.CrossEntropyLoss()
    logits = torch.FloatTensor(pred.predictions[0])
    labels = torch.LongTensor(pred.label_ids)

    loss_value = loss(logits.reshape(-1, num_classes), labels.reshape(-1))
    return {"loss": loss_value}


def is_main_process():
    if not dist.is_torchelastic_launched():
        return True
    return int(os.environ["RANK"]) == 0


def log_on_main(msg: str, logger: logging.Logger, log_level: int = logging.INFO):
    if is_main_process():
        logger.log(log_level, msg)


def has_enough_observations(
    entry: dict, min_length: int = 0, max_missing_prop: float = 1.0
) -> bool:
    """
    Check if the given entry has enough observations in the ``"target"`` attribute.

    Parameters
    ----------
    entry
        The data entry (dictionary) to be tested.
    min_length
        The minimum length the ``"target"`` attribute must have.
    max_missing_prop
        The maximum proportion of missing data allowed in the ``"target"``
        attribute.
    """
    if (
        len(entry["target"]) >= min_length
        and np.isnan(entry["target"]).mean() <= max_missing_prop
    ):
        return True
    return False
