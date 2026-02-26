# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import Trainer

from chronos.chronos2.trainer import Chronos2Trainer


class _DummyModel:
    def __init__(self, device: torch.device, hf_device_map=None):
        self.device = device
        self.hf_device_map = hf_device_map


def test_move_model_to_device_preserves_loaded_cuda_device(monkeypatch):
    """When model is on a single CUDA device, keep that device instead of forcing cuda:0."""
    captured = {}

    def fake_move(self, model, device):
        captured["device"] = device

    monkeypatch.setattr(Trainer, "_move_model_to_device", fake_move)

    trainer = object.__new__(Chronos2Trainer)
    model = _DummyModel(torch.device("cuda:5"))

    Chronos2Trainer._move_model_to_device(trainer, model, torch.device("cuda:0"))

    assert captured["device"] == torch.device("cuda:5")


def test_move_model_to_device_keeps_requested_cpu_device(monkeypatch):
    """CPU fine-tuning should preserve existing Trainer behavior."""
    captured = {}

    def fake_move(self, model, device):
        captured["device"] = device

    monkeypatch.setattr(Trainer, "_move_model_to_device", fake_move)

    trainer = object.__new__(Chronos2Trainer)
    model = _DummyModel(torch.device("cpu"))

    Chronos2Trainer._move_model_to_device(trainer, model, torch.device("cpu"))

    assert captured["device"] == torch.device("cpu")


def test_move_model_to_device_keeps_requested_device_for_hf_device_map(monkeypatch):
    """Do not override device movement for models managed via hf_device_map."""
    captured = {}

    def fake_move(self, model, device):
        captured["device"] = device

    monkeypatch.setattr(Trainer, "_move_model_to_device", fake_move)

    trainer = object.__new__(Chronos2Trainer)
    model = _DummyModel(torch.device("cuda:5"), hf_device_map={"": "cuda:5"})

    Chronos2Trainer._move_model_to_device(trainer, model, torch.device("cuda:0"))

    assert captured["device"] == torch.device("cuda:0")
