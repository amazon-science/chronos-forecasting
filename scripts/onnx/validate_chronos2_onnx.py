#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Validate a Chronos-2 ONNX export against the PyTorch model.

This script exercises the tensor-level ONNX interface used by
export_chronos2_to_onnx.py. It intentionally validates future covariates when
the ONNX graph exposes the `future_covariates` input.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from export_chronos2_to_onnx import Chronos2ONNXWrapper  # noqa: E402
from chronos import Chronos2Pipeline  # noqa: E402


def make_context(batch_size: int, context_length: int, *, missing: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.linspace(0, 10, context_length, dtype=torch.float32)
    rows = []
    for i in range(batch_size):
        trend = (i + 1) * 0.02 * t
        seasonal = torch.sin(t * (1.3 + i * 0.2)) + 0.4 * torch.cos(t * (0.7 + i * 0.1))
        rows.append(trend + seasonal + 0.02 * torch.randn(context_length))

    context = torch.stack(rows)
    attention_mask = torch.ones(batch_size, context_length, dtype=torch.float32)

    if missing:
        context[0, 30:46] = torch.nan
        attention_mask[0, 30:46] = 0.0
        if batch_size > 1:
            context[-1, 211:229] = torch.nan
            attention_mask[-1, 211:229] = 0.0

    return context, attention_mask


def make_future_covariates(batch_size: int, future_length: int, pattern: str, *, missing: bool = False) -> torch.Tensor:
    t = torch.linspace(0, 1, future_length, dtype=torch.float32)

    if pattern == "zeros":
        future_covariates = torch.zeros(batch_size, future_length, dtype=torch.float32)
    elif pattern == "sin":
        future_covariates = torch.stack([torch.sin((i + 1) * torch.pi * t) for i in range(batch_size)])
    elif pattern == "cos":
        future_covariates = torch.stack([torch.cos((i + 1) * torch.pi * t) for i in range(batch_size)])
    elif pattern == "random":
        future_covariates = torch.randn(batch_size, future_length, dtype=torch.float32) * 0.5
    else:
        raise ValueError(f"Unknown covariate pattern: {pattern}")

    if missing:
        future_covariates[0, 8:16] = torch.nan
        if batch_size > 2:
            future_covariates[2, 40:48] = torch.nan

    return future_covariates


def run_case(
    *,
    name: str,
    wrapped_model: Chronos2ONNXWrapper,
    ort_session,
    input_names: set[str],
    batch_size: int,
    context_length: int,
    num_output_patches: int,
    output_patch_size: int,
    group_ids: list[int],
    covariate_pattern: str,
    device: str,
    missing_context: bool = False,
    missing_future_covariates: bool = False,
    rtol: float,
    atol: float,
) -> dict:
    context, attention_mask = make_context(batch_size, context_length, missing=missing_context)
    context = context.to(device)
    attention_mask = attention_mask.to(device)
    group_ids_tensor = torch.tensor(group_ids, dtype=torch.long, device=device)

    include_future_covariates = "future_covariates" in input_names
    future_covariates = None
    if include_future_covariates:
        future_covariates = make_future_covariates(
            batch_size,
            num_output_patches * output_patch_size,
            covariate_pattern,
            missing=missing_future_covariates,
        ).to(device)

    with torch.no_grad():
        pytorch_output = wrapped_model(
            context=context,
            group_ids=group_ids_tensor,
            attention_mask=attention_mask,
            future_covariates=future_covariates,
            num_output_patches=num_output_patches,
        )

    ort_inputs = {
        "context": context.cpu().numpy(),
        "group_ids": group_ids_tensor.cpu().numpy(),
        "attention_mask": attention_mask.cpu().numpy(),
    }
    if include_future_covariates:
        ort_inputs["future_covariates"] = future_covariates.cpu().numpy()
    if "num_output_patches" in input_names:
        ort_inputs["num_output_patches"] = np.array(num_output_patches, dtype=np.int64)

    onnx_output = ort_session.run(None, ort_inputs)[0]
    pytorch_output_np = pytorch_output.cpu().numpy()
    abs_diff = np.abs(pytorch_output_np - onnx_output)

    return {
        "name": name,
        "batch_size": batch_size,
        "group_ids": group_ids,
        "covariate_pattern": covariate_pattern if include_future_covariates else None,
        "missing_context": missing_context,
        "missing_future_covariates": missing_future_covariates if include_future_covariates else None,
        "pytorch_shape": list(pytorch_output_np.shape),
        "onnx_shape": list(onnx_output.shape),
        "max_abs_diff": float(np.nanmax(abs_diff)),
        "mean_abs_diff": float(np.nanmean(abs_diff)),
        "allclose": bool(np.allclose(pytorch_output_np, onnx_output, rtol=rtol, atol=atol, equal_nan=True)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Chronos-2 ONNX parity against PyTorch")
    parser.add_argument("--model_id", type=str, default="amazon/chronos-2", help="HuggingFace model ID or local path")
    parser.add_argument("--onnx_path", type=str, required=True, help="Path to the fixed ONNX model")
    parser.add_argument("--context_length", type=int, default=512, help="Context length used during export")
    parser.add_argument("--num_output_patches", type=int, default=4, help="Number of output patches used during export")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="PyTorch device")
    parser.add_argument("--rtol", type=float, default=1e-4, help="Relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance")
    parser.add_argument("--report_path", type=str, default=None, help="Optional JSON report output path")
    args = parser.parse_args()

    torch.manual_seed(123)
    np.random.seed(123)

    pipeline = Chronos2Pipeline.from_pretrained(args.model_id, device_map=args.device)
    model = pipeline.model.eval()
    wrapped_model = Chronos2ONNXWrapper(model).eval()

    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.device == "cuda" else ["CPUExecutionProvider"]
    ort_session = ort.InferenceSession(args.onnx_path, providers=providers)
    input_names = {input_.name for input_ in ort_session.get_inputs()}

    output_patch_size = model.chronos_config.output_patch_size
    cases = [
        dict(name="batch1_zeros", batch_size=1, group_ids=[0], covariate_pattern="zeros"),
        dict(name="batch2_shared_group_sin", batch_size=2, group_ids=[0, 0], covariate_pattern="sin"),
        dict(name="batch3_mixed_groups_cos", batch_size=3, group_ids=[0, 1, 0], covariate_pattern="cos"),
        dict(name="batch4_distinct_random", batch_size=4, group_ids=[0, 1, 2, 3], covariate_pattern="random"),
        dict(
            name="batch3_missing_context_sin",
            batch_size=3,
            group_ids=[0, 1, 0],
            covariate_pattern="sin",
            missing_context=True,
        ),
        dict(
            name="batch3_missing_future_cos",
            batch_size=3,
            group_ids=[0, 1, 0],
            covariate_pattern="cos",
            missing_future_covariates=True,
        ),
    ]

    results = [
        run_case(
            wrapped_model=wrapped_model,
            ort_session=ort_session,
            input_names=input_names,
            context_length=args.context_length,
            num_output_patches=args.num_output_patches,
            output_patch_size=output_patch_size,
            device=args.device,
            rtol=args.rtol,
            atol=args.atol,
            **case,
        )
        for case in cases
    ]

    report = {
        "model_id": args.model_id,
        "onnx_path": args.onnx_path,
        "providers": ort_session.get_providers(),
        "inputs": [(input_.name, input_.type, input_.shape) for input_ in ort_session.get_inputs()],
        "outputs": [(output.name, output.type, output.shape) for output in ort_session.get_outputs()],
        "rtol": args.rtol,
        "atol": args.atol,
        "all_cases_passed": all(result["allclose"] for result in results),
        "cases": results,
    }

    report_json = json.dumps(report, indent=2)
    print(report_json)

    if args.report_path:
        Path(args.report_path).write_text(report_json)

    return 0 if report["all_cases_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
