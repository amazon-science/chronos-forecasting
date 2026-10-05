#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Export Chronos-2 models to ONNX format.

This script:
1. Loads a pretrained Chronos-2 model
2. Exports it to ONNX format with dynamic batch size
3. Validates the ONNX export by comparing outputs with PyTorch
4. Optionally quantizes the model for smaller size

Usage:
    python export_chronos2_to_onnx.py \
        --model_id amazon/chronos-2 \
        --output_dir ./chronos2-onnx \
        --validate

Requirements:
    pip install torch onnx onnxruntime onnxscript transformers chronos-forecasting
"""

import argparse
import logging
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import numpy as np

from chronos import Chronos2Pipeline

# Register custom ONNX symbolic functions for operations that aren't properly mapped
from torch.onnx import register_custom_op_symbolic


def asinh_symbolic(g, input):
    """Custom ONNX symbolic function for asinh (arcsinh)."""
    return g.op("Asinh", input)


def sinh_symbolic(g, input):
    """Custom ONNX symbolic function for sinh."""
    return g.op("Sinh", input)


# Register the symbolic functions for opset 9+
register_custom_op_symbolic("aten::asinh", asinh_symbolic, 9)
register_custom_op_symbolic("aten::sinh", sinh_symbolic, 9)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class Chronos2ONNXWrapper(nn.Module):
    """
    Wrapper around Chronos2Model to handle ONNX export.

    This wrapper simplifies the input/output interface for ONNX export
    by flattening the input dictionary structure.
    """

    def __init__(self, chronos2_model):
        super().__init__()
        self.model = chronos2_model

    def forward(
        self,
        context: torch.Tensor,
        group_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        future_covariates: torch.Tensor | None = None,
        num_output_patches: int = 1,
    ):
        """
        Forward pass compatible with ONNX export.

        Args:
            context: Historical context tensor of shape (batch_size, context_length)
            group_ids: Group IDs tensor of shape (batch_size,)
            attention_mask: Optional attention mask of shape (batch_size, context_length)
            future_covariates: Optional future covariates of shape (batch_size, future_length)
            num_output_patches: Number of output patches to generate

        Returns:
            quantile_preds: Tensor of shape (batch_size, num_quantiles, prediction_length)
        """
        kwargs = {
            "context": context,
            "group_ids": group_ids,
            "num_output_patches": num_output_patches,
        }

        if attention_mask is not None:
            kwargs["context_mask"] = attention_mask

        if future_covariates is not None:
            kwargs["future_covariates"] = future_covariates

        # Run model forward pass
        outputs = self.model(**kwargs)

        # Return only the quantile predictions (drop loss and attention weights)
        return outputs.quantile_preds


def create_dummy_inputs(
    batch_size: int = 2,
    context_length: int = 512,
    num_output_patches: int = 1,
    include_future_covariates: bool = False,
    output_patch_size: int = 64,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Create dummy inputs for ONNX export.

    Args:
        batch_size: Batch size
        context_length: Length of historical context
        num_output_patches: Number of output patches
        include_future_covariates: Whether to include future covariates
        output_patch_size: Size of each output patch
        device: Device to create tensors on

    Returns:
        Dictionary of dummy inputs
    """
    dummy_inputs = {
        "context": torch.randn(batch_size, context_length, device=device, dtype=torch.float32),
        "group_ids": torch.arange(batch_size, device=device, dtype=torch.long),
        "attention_mask": torch.ones(batch_size, context_length, device=device, dtype=torch.float32),
        "num_output_patches": num_output_patches,
    }

    if include_future_covariates:
        future_length = num_output_patches * output_patch_size
        dummy_inputs["future_covariates"] = torch.randn(batch_size, future_length, device=device, dtype=torch.float32)

    return dummy_inputs


def export_to_onnx(
    model_id: str,
    output_dir: Path,
    opset_version: int = 17,
    use_fp16: bool = False,
    include_future_covariates: bool = True,
    context_length: int | None = None,
    num_output_patches: int = 4,
    device: str = None,
    onnx_filename: str = "model_raw.onnx",
) -> Path:
    """
    Export Chronos-2 model to ONNX format.

    Args:
        model_id: HuggingFace model ID or local path
        output_dir: Directory to save ONNX model
        opset_version: ONNX opset version (17 recommended for best compatibility)
        use_fp16: Whether to use FP16 precision
        include_future_covariates: Whether to support future covariates in export
        context_length: Context length to trace into the model. If omitted, uses min(512, model context length).
        num_output_patches: Number of output patches to trace into the model.
        device: Device to use ('cuda' or 'cpu')
        onnx_filename: File name for the raw exported ONNX model

    Returns:
        Path to exported ONNX model
    """
    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading Chronos-2 model from {model_id}")

    # Load the pipeline and extract the model
    # Official model is now available at: https://huggingface.co/amazon/chronos-2
    pipeline = Chronos2Pipeline.from_pretrained(model_id, device_map=device)

    model = pipeline.model
    config = model.config
    chronos_config = model.chronos_config

    logger.info(
        f"Model config: {config.model_type}, d_model={config.d_model}, "
        f"num_layers={config.num_layers}, num_heads={config.num_heads}"
    )
    logger.info(
        f"Chronos config: context_length={chronos_config.context_length}, "
        f"output_patch_size={chronos_config.output_patch_size}, "
        f"quantiles={chronos_config.quantiles}"
    )

    # Set model to eval mode
    model.eval()

    # Convert to FP16 if requested
    if use_fp16:
        logger.info("Converting model to FP16")
        model = model.half()

    # Wrap model for ONNX export
    wrapped_model = Chronos2ONNXWrapper(model)
    wrapped_model.eval()

    # Create dummy inputs
    batch_size = 2
    if context_length is None:
        context_length = min(512, chronos_config.context_length)

    if context_length <= 0:
        raise ValueError(f"context_length must be positive, found {context_length}")

    if context_length > chronos_config.context_length:
        raise ValueError(
            f"context_length={context_length} exceeds model context length {chronos_config.context_length}"
        )

    if num_output_patches <= 0:
        raise ValueError(f"num_output_patches must be positive, found {num_output_patches}")

    if num_output_patches > chronos_config.max_output_patches:
        raise ValueError(
            f"num_output_patches={num_output_patches} exceeds model maximum "
            f"{chronos_config.max_output_patches}"
        )

    dummy_inputs = create_dummy_inputs(
        batch_size=batch_size,
        context_length=context_length,
        num_output_patches=num_output_patches,
        include_future_covariates=include_future_covariates,
        output_patch_size=chronos_config.output_patch_size,
        device=device,
    )

    prediction_length = num_output_patches * chronos_config.output_patch_size

    # Keep sequence lengths fixed because legacy ONNX export cannot lower the patching path
    # when the patched sequence dimension is symbolic. Batch remains dynamic.
    dynamic_axes = {
        "context": {0: "batch_size"},
        "group_ids": {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "quantile_preds": {0: "batch_size"},
    }

    if include_future_covariates:
        dynamic_axes["future_covariates"] = {0: "batch_size"}

    # Prepare ONNX export args based on whether future_covariates are included
    if include_future_covariates:
        input_names = ["context", "group_ids", "attention_mask", "future_covariates"]
        args = (
            dummy_inputs["context"],
            dummy_inputs["group_ids"],
            dummy_inputs["attention_mask"],
            dummy_inputs["future_covariates"],
            dummy_inputs["num_output_patches"],
        )
    else:
        input_names = ["context", "group_ids", "attention_mask"]
        args = (
            dummy_inputs["context"],
            dummy_inputs["group_ids"],
            dummy_inputs["attention_mask"],
            None,  # No future_covariates
            dummy_inputs["num_output_patches"],
        )

    output_names = ["quantile_preds"]

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / onnx_filename

    logger.info(f"Exporting model to ONNX format at {onnx_path}")
    logger.info(f"Dynamic axes: {dynamic_axes}")
    logger.info(f"Fixed context_length={context_length}, prediction_length={prediction_length}")

    # Export to ONNX
    try:
        with torch.no_grad():
            # Skip dynamo exporter when using covariates (has dtype issues with embeddings)
            # Always use legacy exporter for now as it's more reliable
            use_dynamo = False  # Disabled due to dtype issues with Gather ops in embeddings

            if use_dynamo and not include_future_covariates:
                # Try new dynamo-based exporter first (supports more ops like nanmean)
                try:
                    torch.onnx.export(
                        wrapped_model,
                        args,
                        str(onnx_path),
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_axes=dynamic_axes,
                        dynamo=True,  # Use new PyTorch 2.x+ exporter
                        verbose=False,
                    )
                    logger.info("Used dynamo-based ONNX exporter")
                except Exception as dynamo_error:
                    logger.warning(f"Dynamo exporter failed ({dynamo_error}), trying legacy exporter...")
                    use_dynamo = False

            if not use_dynamo:
                # Use legacy exporter (more reliable for embeddings)
                logger.info("Using legacy TorchScript-based ONNX exporter")
                torch.onnx.export(
                    wrapped_model,
                    args,
                    str(onnx_path),
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    export_params=True,
                    dynamo=False,
                    verbose=False,
                )
                logger.info("Used legacy TorchScript-based ONNX exporter")
        logger.info(f"Successfully exported model to {onnx_path}")
    except Exception as e:
        logger.error(f"Failed to export model to ONNX: {e}")
        raise

    # Save config files
    config_path = output_dir / "config.json"
    config.save_pretrained(output_dir)
    logger.info(f"Saved config to {config_path}")

    # Save generation config if it exists
    if hasattr(pipeline, "generation_config"):
        generation_config_path = output_dir / "generation_config.json"
        pipeline.generation_config.save_pretrained(output_dir)
        logger.info(f"Saved generation config to {generation_config_path}")

    return onnx_path


def fix_onnx_export(raw_onnx_path: Path, fixed_onnx_path: Path) -> Path:
    """
    Run the mandatory Chronos-2 ONNX post-processing pass.

    The legacy exporter emits some Gather indices with the wrong dtype. ONNX Runtime
    rejects the raw graph until those indices are cast back to int64.
    """
    from fix_onnx_model import fix_gather_indices

    logger.info("Fixing exported ONNX graph")
    logger.info(f"  Raw:   {raw_onnx_path}")
    logger.info(f"  Fixed: {fixed_onnx_path}")

    fix_gather_indices(str(raw_onnx_path), str(fixed_onnx_path), make_dynamic=False)
    return fixed_onnx_path


def quantize_model(onnx_path: Path) -> Path:
    """
    Quantize the ONNX model to INT8.

    Args:
        onnx_path: Path to the FP32 ONNX model

    Returns:
        Path to the quantized model
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        logger.error("onnxruntime not installed. Install with: pip install onnxruntime")
        raise

    quantized_path = onnx_path.parent / "model_quantized.onnx"

    logger.info("Quantizing model to INT8...")
    logger.info(f"  Input:  {onnx_path}")
    logger.info(f"  Output: {quantized_path}")

    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(quantized_path),
        weight_type=QuantType.QInt8,
    )

    # Compare sizes
    original_size = onnx_path.stat().st_size / (1024**2)  # MB
    quantized_size = quantized_path.stat().st_size / (1024**2)  # MB
    reduction = (1 - quantized_size / original_size) * 100

    logger.info(f"  Original:  {original_size:.1f} MB")
    logger.info(f"  Quantized: {quantized_size:.1f} MB")
    logger.info(f"  Reduction: {reduction:.1f}%")

    return quantized_path


def setup_transformersjs_structure(output_dir: Path):
    """
    Create transformers.js-compatible directory structure.

    Creates:
    - onnx/ directory with symlinks to model files
    - generation_config.json if missing
    """
    import json
    import os

    logger.info("Setting up transformers.js directory structure...")

    # Create onnx/ subdirectory
    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(exist_ok=True)

    # Create symlinks for encoder/decoder (transformers.js expects T5-style split)
    output_dir / "model.onnx"
    encoder_link = onnx_dir / "encoder_model.onnx"
    decoder_link = onnx_dir / "decoder_model_merged.onnx"

    # Remove existing symlinks if they exist
    if encoder_link.exists() or encoder_link.is_symlink():
        encoder_link.unlink()
    if decoder_link.exists() or decoder_link.is_symlink():
        decoder_link.unlink()

    # Create new symlinks
    os.symlink("../model.onnx", encoder_link)
    os.symlink("../model.onnx", decoder_link)

    logger.info(f"  Created {encoder_link}")
    logger.info(f"  Created {decoder_link}")

    # Create minimal generation_config.json if missing
    generation_config_path = output_dir / "generation_config.json"
    if not generation_config_path.exists():
        generation_config = {"_from_model_config": True, "transformers_version": "4.36.0"}
        with open(generation_config_path, "w") as f:
            json.dump(generation_config, f, indent=2)
        logger.info(f"  Created {generation_config_path}")


def generate_readme(output_dir: Path, model_id: str, quantized: bool = False):
    """
    Generate README.md with model card for Hub.

    Args:
        output_dir: Output directory
        model_id: Original model ID
        quantized: Whether quantized model is included
    """
    import json

    # Load config to get model details
    config_path = output_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    chronos_config = config.get("chronos_config", {})
    output_patch_size = chronos_config.get("output_patch_size", 16)
    quantiles = chronos_config.get("quantiles", [])
    quantized_line = ""
    if quantized and (output_dir / "model_quantized.onnx").exists():
        quantized_line = (
            f"- `model_quantized.onnx` - INT8 dynamic-quantized ONNX model "
            f"({(output_dir / 'model_quantized.onnx').stat().st_size / (1024**2):.1f} MB)\n"
        )

    readme_content = f"""---
library_name: onnx
tags:
  - time-series
  - forecasting
  - chronos
  - onnx
pipeline_tag: time-series-forecasting
---

# Chronos-2 ONNX

This is a tensor-level ONNX export of the [Chronos-2]({model_id}) time series forecasting model.

## Model Details

- **Model Type:** Time Series Forecasting
- **Architecture:** Chronos-2 encoder-decoder with patching
- **Maximum Model Context Length:** {chronos_config.get("context_length", 8192)} timesteps
- **Input Patch Size:** {chronos_config.get("input_patch_size", 16)} timesteps
- **Output Patch Size:** {output_patch_size} timesteps
- **Quantile Levels:** {len(quantiles)} levels
- **Model Dimension:** {config.get("d_model", 768)}
- **Layers:** {config.get("num_layers", 12)}
- **Attention Heads:** {config.get("num_heads", 12)}

## Files

- `model.onnx` - FP32 ONNX model ({(output_dir / "model.onnx").stat().st_size / (1024**2):.1f} MB)
{quantized_line.rstrip()}
- `config.json` - Model configuration
- `generation_config.json` - Generation parameters

## Usage

Run the exported model with ONNX Runtime:

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
input_names = {{input_.name for input_ in session.get_inputs()}}

batch_size = 1
context_length = 512
num_output_patches = 4
prediction_length = num_output_patches * {output_patch_size}

inputs = {{
    "context": np.random.randn(batch_size, context_length).astype(np.float32),
    "group_ids": np.arange(batch_size, dtype=np.int64),
    "attention_mask": np.ones((batch_size, context_length), dtype=np.float32),
}}

if "future_covariates" in input_names:
    inputs["future_covariates"] = np.random.randn(batch_size, prediction_length).astype(np.float32)

if "num_output_patches" in input_names:
    inputs["num_output_patches"] = np.array(num_output_patches, dtype=np.int64)

quantile_preds = session.run(None, inputs)[0]
print(quantile_preds.shape)
```

## Tensor Interface

- `context`: float32 tensor shaped `[batch, context_length]`.
- `group_ids`: int64 tensor shaped `[batch]`. Equal IDs allow time series in the same group to attend to each other.
- `attention_mask`: float32 tensor shaped `[batch, context_length]`, with `1` for observed context positions and `0` for masked positions.
- `future_covariates`: optional float32 tensor shaped `[batch, prediction_length]` when the model was exported with future covariates.
- `num_output_patches`: optional int64 scalar if the exported graph exposes it. Use the same value passed at export time.
- `quantile_preds`: float32 output shaped `[batch, num_quantiles, prediction_length]`.

## Limitations

- Batch size is dynamic.
- Context length, future covariate length, and prediction length are fixed by the traced export.
- The default export traces `context_length=512` and `num_output_patches=4`, which gives a `{4 * output_patch_size}` step horizon for this model.
- This artifact exposes the model's tensor interface. Any serving API, preprocessing wrapper, or browser runtime integration is separate from the export.
- Validate PyTorch vs ONNX parity with `validate_chronos2_onnx.py` after export and before relying on a quantized model.

## Citation

```bibtex
@article{{ansari2024chronos,
  title={{Chronos: Learning the Language of Time Series}},
  author={{Ansari, Abdul Fatir and others}},
  journal={{arXiv preprint arXiv:2403.07815}},
  year={{2024}}
}}
```

## License

Apache 2.0

## Links

- [Chronos-2 Paper](https://arxiv.org/abs/2403.07815)
- [Chronos GitHub](https://github.com/amazon-science/chronos-forecasting)
"""

    readme_path = output_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)

    logger.info(f"  Generated {readme_path}")


def push_to_hub(output_dir: Path, repo_id: str, private: bool = False):
    """
    Push the model to HuggingFace Hub.

    Args:
        output_dir: Directory containing the model files
        repo_id: Hub repository ID (e.g., 'username/chronos-2-onnx')
        private: Whether to make the repository private
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        logger.error("huggingface_hub not installed. Install with: pip install huggingface-hub")
        raise

    logger.info(f"\nPushing to HuggingFace Hub: {repo_id}")

    api = HfApi()

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, private=private, exist_ok=True)
        logger.info(f"  Repository created/verified: https://huggingface.co/{repo_id}")
    except Exception as e:
        logger.warning(f"  Could not create repo: {e}")

    # Upload all files
    logger.info("  Uploading files...")

    files_to_upload = [
        "model.onnx",
        "config.json",
        "generation_config.json",
        "README.md",
    ]

    # Add quantized model if it exists
    if (output_dir / "model_quantized.onnx").exists():
        files_to_upload.append("model_quantized.onnx")

    # Upload onnx/ directory
    for file in files_to_upload:
        file_path = output_dir / file
        if file_path.exists():
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=file,
                repo_id=repo_id,
                repo_type="model",
            )
            logger.info(f"    ✓ {file}")

    # Upload onnx/ directory symlinks (as actual files)
    onnx_dir = output_dir / "onnx"
    if onnx_dir.exists():
        for file in ["encoder_model.onnx", "decoder_model_merged.onnx"]:
            src_path = output_dir / "model.onnx"
            if src_path.exists():
                api.upload_file(
                    path_or_fileobj=str(src_path),
                    path_in_repo=f"onnx/{file}",
                    repo_id=repo_id,
                    repo_type="model",
                )
                logger.info(f"    ✓ onnx/{file}")

    logger.info(f"\n✓ Successfully pushed to: https://huggingface.co/{repo_id}")


def validate_onnx_export(
    onnx_path: Path,
    model_id: str,
    device: str = None,
    include_future_covariates: bool = True,
    context_length: int = 512,
    num_output_patches: int = 4,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> bool:
    """
    Validate ONNX export by comparing outputs with PyTorch model.

    Args:
        onnx_path: Path to ONNX model
        model_id: Original model ID
        device: Device to use
        include_future_covariates: Whether to validate the future covariate input
        context_length: Context length expected by the exported model
        num_output_patches: Number of output patches expected by the exported model
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison

    Returns:
        True if validation passes
    """
    logger.info("Validating ONNX export...")

    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load PyTorch model
    # Official model is now available at: https://huggingface.co/amazon/chronos-2
    pipeline = Chronos2Pipeline.from_pretrained(model_id, device_map=device)

    model = pipeline.model
    model.eval()

    # Load ONNX model
    import onnxruntime as ort

    logger.info(f"Loading ONNX model from {onnx_path}")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    ort_session = ort.InferenceSession(str(onnx_path), providers=providers)
    input_names = {input_.name for input_ in ort_session.get_inputs()}

    batch_size = 4

    dummy_inputs = create_dummy_inputs(
        batch_size=batch_size,
        context_length=context_length,
        num_output_patches=num_output_patches,
        include_future_covariates=include_future_covariates,
        output_patch_size=model.chronos_config.output_patch_size,
        device=device,
    )

    # Run PyTorch inference
    logger.info("Running PyTorch inference...")
    with torch.no_grad():
        wrapped_model = Chronos2ONNXWrapper(model)
        pytorch_output = wrapped_model(
            context=dummy_inputs["context"],
            group_ids=dummy_inputs["group_ids"],
            attention_mask=dummy_inputs["attention_mask"],
            future_covariates=dummy_inputs.get("future_covariates"),
            num_output_patches=dummy_inputs["num_output_patches"],
        )

    # Run ONNX inference. Some PyTorch versions expose num_output_patches as a
    # scalar graph input even when it is passed as a Python int during tracing.
    logger.info("Running ONNX inference...")
    ort_inputs = {
        "context": dummy_inputs["context"].cpu().numpy(),
        "group_ids": dummy_inputs["group_ids"].cpu().numpy(),
        "attention_mask": dummy_inputs["attention_mask"].cpu().numpy(),
    }
    if "future_covariates" in input_names:
        ort_inputs["future_covariates"] = dummy_inputs["future_covariates"].cpu().numpy()
    if "num_output_patches" in input_names:
        ort_inputs["num_output_patches"] = np.array(dummy_inputs["num_output_patches"], dtype=np.int64)

    onnx_output = ort_session.run(None, ort_inputs)[0]

    # Compare outputs
    pytorch_output_np = pytorch_output.cpu().numpy()

    logger.info(f"PyTorch output shape: {pytorch_output_np.shape}")
    logger.info(f"ONNX output shape: {onnx_output.shape}")

    # Check shapes match
    if pytorch_output_np.shape != onnx_output.shape:
        logger.error(f"Output shapes don't match! PyTorch: {pytorch_output_np.shape}, ONNX: {onnx_output.shape}")
        return False

    # Check values match
    max_diff = np.abs(pytorch_output_np - onnx_output).max()
    mean_diff = np.abs(pytorch_output_np - onnx_output).mean()

    logger.info(f"Max absolute difference: {max_diff:.6f}")
    logger.info(f"Mean absolute difference: {mean_diff:.6f}")

    if np.allclose(pytorch_output_np, onnx_output, rtol=rtol, atol=atol):
        logger.info("✓ Validation PASSED: ONNX output matches PyTorch output")
        return True
    else:
        logger.error("✗ Validation FAILED: ONNX output doesn't match PyTorch output")
        logger.error(f"Relative tolerance: {rtol}, Absolute tolerance: {atol}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export Chronos-2 model to ONNX format")
    parser.add_argument(
        "--model_id",
        type=str,
        default="amazon/chronos-2",
        help="HuggingFace model ID or local path (e.g., 'amazon/chronos-2')",
    )
    parser.add_argument("--output_dir", type=str, default="./chronos2-onnx", help="Output directory for ONNX model")
    parser.add_argument("--opset_version", type=int, default=17, help="ONNX opset version (default: 17)")
    parser.add_argument(
        "--context_length",
        type=int,
        default=512,
        help="Fixed context length to export (default: 512; batch size remains dynamic)",
    )
    parser.add_argument(
        "--num_output_patches",
        type=int,
        default=4,
        help="Number of output patches to export (default: 4; with Chronos-2 output_patch_size=16 this gives 64 steps)",
    )
    parser.add_argument("--fp16", action="store_true", help="Export model in FP16 precision")
    parser.add_argument(
        "--validate", action="store_true", help="Validate ONNX export by comparing with PyTorch outputs"
    )
    parser.add_argument(
        "--no_future_covariates", action="store_true", help="Don't include future covariates support in export"
    )
    parser.add_argument(
        "--device", type=str, default=None, choices=["cpu", "cuda"], help="Device to use (default: auto-detect)"
    )
    parser.add_argument(
        "--no_fix_onnx",
        action="store_true",
        help="Skip the mandatory post-export ONNX graph fix pass (not recommended; raw graph may not load in ONNX Runtime)",
    )
    parser.add_argument(
        "--keep_raw_onnx",
        action="store_true",
        help="Keep model_raw.onnx after writing the fixed model.onnx",
    )
    parser.add_argument(
        "--setup_transformersjs",
        action="store_true",
        help="Also create the legacy onnx/ symlink layout used by the original browser demo.",
    )
    parser.add_argument("--quantize", action="store_true", help="Quantize the model to INT8 after export")
    parser.add_argument(
        "--push_to_hub",
        type=str,
        default=None,
        help="Push the exported model to HuggingFace Hub (e.g., 'username/chronos-2-onnx')",
    )
    parser.add_argument("--private", action="store_true", help="Make the Hub repository private")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    try:
        # Export model
        logger.info("=" * 60)
        logger.info("Chronos-2 ONNX Export Pipeline")
        logger.info("=" * 60 + "\n")

        raw_or_final_onnx_path = export_to_onnx(
            model_id=args.model_id,
            output_dir=output_dir,
            opset_version=args.opset_version,
            use_fp16=args.fp16,
            include_future_covariates=not args.no_future_covariates,
            context_length=args.context_length,
            num_output_patches=args.num_output_patches,
            device=args.device,
            onnx_filename="model.onnx" if args.no_fix_onnx else "model_raw.onnx",
        )

        if args.no_fix_onnx:
            onnx_path = raw_or_final_onnx_path
            logger.warning("Skipping ONNX fix pass; the raw graph may not load in ONNX Runtime")
        else:
            onnx_path = fix_onnx_export(raw_or_final_onnx_path, output_dir / "model.onnx")
            if not args.keep_raw_onnx:
                raw_or_final_onnx_path.unlink(missing_ok=True)
                logger.info(f"Removed intermediate raw ONNX model: {raw_or_final_onnx_path}")

        # Validate if requested
        if args.validate:
            logger.info("\n" + "=" * 60)
            logger.info("Validation")
            logger.info("=" * 60 + "\n")

            validation_passed = validate_onnx_export(
                onnx_path=onnx_path,
                model_id=args.model_id,
                device=args.device,
                include_future_covariates=not args.no_future_covariates,
                context_length=args.context_length,
                num_output_patches=args.num_output_patches,
            )

            if not validation_passed:
                logger.warning("Validation failed, but ONNX model was still exported")
                return 1

        # Quantize if requested
        quantized_path = None
        if args.quantize:
            logger.info("\n" + "=" * 60)
            logger.info("Quantization")
            logger.info("=" * 60 + "\n")

            quantized_path = quantize_model(onnx_path)

        if args.setup_transformersjs:
            # Setup transformers.js directory structure
            logger.info("\n" + "=" * 60)
            logger.info("transformers.js Setup")
            logger.info("=" * 60 + "\n")

            setup_transformersjs_structure(output_dir)

        # Generate README
        logger.info("\n" + "=" * 60)
        logger.info("README Generation")
        logger.info("=" * 60 + "\n")

        generate_readme(output_dir, args.model_id, quantized=args.quantize)

        # Push to Hub if requested
        if args.push_to_hub:
            logger.info("\n" + "=" * 60)
            logger.info("Hub Upload")
            logger.info("=" * 60 + "\n")

            push_to_hub(output_dir, args.push_to_hub, private=args.private)

        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("Export Complete!")
        logger.info("=" * 60)
        logger.info(f"  ONNX model: {onnx_path}")
        if quantized_path:
            logger.info(f"  Quantized:  {quantized_path}")
        logger.info(f"  Config:     {output_dir / 'config.json'}")
        logger.info(f"  README:     {output_dir / 'README.md'}")
        if args.push_to_hub:
            logger.info(f"  Hub URL:    https://huggingface.co/{args.push_to_hub}")
        logger.info("=" * 60 + "\n")

        return 0

    except Exception as e:
        logger.error(f"Export failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
