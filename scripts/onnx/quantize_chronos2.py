#!/usr/bin/env python3
"""
Quantize Chronos-2 ONNX model to reduce size and improve inference speed.

This script quantizes the ONNX model from FP32 to INT8, reducing model size
by approximately 75% while maintaining good accuracy.

Usage:
    python quantize_chronos2.py \
        --input chronos2-onnx/model.onnx \
        --output chronos2-onnx/model_quantized.onnx \
        --mode dynamic

Quantization Modes:
    - dynamic: Dynamic quantization (fastest, best compatibility)
    - static: Static quantization (requires calibration data, best accuracy)
    - qat: Quantization-aware training (requires retraining)
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _static_dim(value, default: int) -> int:
    return value if isinstance(value, int) and value > 0 else default


def _input_dim(session, input_name: str, axis: int, default: int) -> int:
    for input_ in session.get_inputs():
        if input_.name == input_name and len(input_.shape) > axis:
            return _static_dim(input_.shape[axis], default)
    return default


def _output_dim(session, axis: int, default: int) -> int:
    outputs = session.get_outputs()
    if outputs and len(outputs[0].shape) > axis:
        return _static_dim(outputs[0].shape[axis], default)
    return default


def _infer_num_output_patches(model_path: str, prediction_length: int, default: int = 4) -> int:
    config_path = Path(model_path).parent / "config.json"
    if not config_path.exists():
        return default

    with open(config_path) as f:
        config = json.load(f)

    output_patch_size = config.get("chronos_config", {}).get("output_patch_size")
    if not isinstance(output_patch_size, int) or output_patch_size <= 0:
        return default

    return max(1, prediction_length // output_patch_size)


def _make_test_inputs(
    input_names: set[str],
    batch_size: int,
    context_length: int,
    prediction_length: int,
    num_output_patches: int,
) -> dict[str, np.ndarray]:
    inputs = {
        "context": np.random.randn(batch_size, context_length).astype(np.float32),
        "group_ids": np.arange(batch_size, dtype=np.int64),
        "attention_mask": np.ones((batch_size, context_length), dtype=np.float32),
    }
    if "future_covariates" in input_names:
        inputs["future_covariates"] = np.random.randn(batch_size, prediction_length).astype(np.float32)
    if "num_output_patches" in input_names:
        inputs["num_output_patches"] = np.array(num_output_patches, dtype=np.int64)

    return inputs


def dynamic_quantization(model_path: str, output_path: str):
    """
    Apply dynamic quantization to the ONNX model.

    Dynamic quantization converts weights to INT8 at export time and
    activations to INT8 dynamically at runtime.

    Pros:
    - No calibration data needed
    - 4x smaller model size
    - Faster inference on CPU
    - Good accuracy (typically <1% loss)

    Cons:
    - Activations still computed in FP32 then converted
    - Less speedup than static quantization
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    logger.info(f"Loading model from {model_path}")

    logger.info("Applying dynamic quantization...")
    logger.info("  - Weight type: INT8")
    logger.info("  - Activation type: INT8 (dynamic)")

    quantize_dynamic(
        model_input=model_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
    )

    logger.info(f"Quantized model saved to {output_path}")


def static_quantization(model_path: str, output_path: str, calibration_data_path: str = None):
    """
    Apply static quantization to the ONNX model.

    Static quantization requires calibration data to determine optimal
    quantization parameters for both weights and activations.

    Pros:
    - Best inference speed
    - Smallest model size
    - Activations also quantized

    Cons:
    - Requires representative calibration data
    - More complex setup
    - Potential accuracy loss if calibration data not representative
    """
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader

    logger.info(f"Loading model from {model_path}")

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_names = {input_.name for input_ in session.get_inputs()}
    context_length = _input_dim(session, "context", 1, 512)
    prediction_length = _input_dim(session, "future_covariates", 1, _output_dim(session, 2, 64))
    num_output_patches = _infer_num_output_patches(model_path, prediction_length)

    # Create calibration data reader
    if calibration_data_path:
        logger.info(f"Loading calibration data from {calibration_data_path}")
        # Custom calibration data reader would go here
        raise NotImplementedError("Custom calibration data reader not implemented yet")
    else:
        logger.info("Generating synthetic calibration data...")

        class SyntheticCalibrationDataReader(CalibrationDataReader):
            def __init__(self, num_samples=100):
                self.num_samples = num_samples
                self.current_sample = 0
                self.batch_size = 1

            def get_next(self):
                if self.current_sample >= self.num_samples:
                    return None

                self.current_sample += 1

                return _make_test_inputs(
                    input_names=input_names,
                    batch_size=self.batch_size,
                    context_length=context_length,
                    prediction_length=prediction_length,
                    num_output_patches=num_output_patches,
                )

        calibration_data_reader = SyntheticCalibrationDataReader()

    logger.info("Applying static quantization...")
    logger.info("  - Weight type: INT8")
    logger.info("  - Activation type: INT8 (static)")
    logger.info("  - Calibration samples: 100")

    quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=calibration_data_reader,
        quant_format=QuantType.QInt8,
    )

    logger.info(f"Quantized model saved to {output_path}")


def compare_models(original_path: str, quantized_path: str):
    """Compare original and quantized model sizes."""

    original_size = Path(original_path).stat().st_size / (1024**2)  # MB
    quantized_size = Path(quantized_path).stat().st_size / (1024**2)  # MB

    reduction = (1 - quantized_size / original_size) * 100

    logger.info(f"\n{'=' * 60}")
    logger.info("Model Size Comparison:")
    logger.info(f"  Original:  {original_size:.1f} MB")
    logger.info(f"  Quantized: {quantized_size:.1f} MB")
    logger.info(f"  Reduction: {reduction:.1f}%")
    logger.info(f"{'=' * 60}\n")


def validate_quantized_model(model_path: str):
    """Validate the quantized model can be loaded and run."""

    logger.info("Validating quantized model...")

    try:
        import onnxruntime as ort

        # Load model
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        input_names = {input_.name for input_ in session.get_inputs()}

        # Create test input
        batch_size = 1
        context_length = _input_dim(session, "context", 1, 512)
        prediction_length = _input_dim(session, "future_covariates", 1, _output_dim(session, 2, 64))
        num_output_patches = _infer_num_output_patches(model_path, prediction_length)

        inputs = _make_test_inputs(
            input_names=input_names,
            batch_size=batch_size,
            context_length=context_length,
            prediction_length=prediction_length,
            num_output_patches=num_output_patches,
        )

        # Run inference
        logger.info("  Running test inference...")
        outputs = session.run(None, inputs)

        logger.info("  ✓ Inference successful!")
        logger.info(f"  Output shape: {outputs[0].shape}")
        logger.info(f"  Output dtype: {outputs[0].dtype}")

        return True

    except Exception as e:
        logger.error(f"  ✗ Validation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Quantize Chronos-2 ONNX model")
    parser.add_argument("--input", type=str, default="chronos2-onnx/model.onnx", help="Input ONNX model path")
    parser.add_argument(
        "--output", type=str, default="chronos2-onnx/model_quantized.onnx", help="Output quantized model path"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="dynamic",
        choices=["dynamic", "static"],
        help="Quantization mode (dynamic or static)",
    )
    parser.add_argument(
        "--calibration_data", type=str, default=None, help="Path to calibration data (for static quantization)"
    )
    parser.add_argument("--validate", action="store_true", help="Validate quantized model after export")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Chronos-2 ONNX Model Quantization")
    logger.info("=" * 60)

    # Check if onnxruntime is installed
    try:
        import onnxruntime

        logger.info(f"ONNX Runtime version: {onnxruntime.__version__}")
    except ImportError:
        logger.error("onnxruntime not installed. Install with: pip install onnxruntime")
        return 1

    # Run quantization
    try:
        if args.mode == "dynamic":
            dynamic_quantization(args.input, args.output)
        elif args.mode == "static":
            static_quantization(args.input, args.output, args.calibration_data)

        # Compare sizes
        compare_models(args.input, args.output)

        # Validate if requested
        if args.validate:
            if validate_quantized_model(args.output):
                logger.info("✓ Quantization completed successfully!")
                return 0
            else:
                logger.warning("⚠ Quantization completed but validation failed")
                return 1
        else:
            logger.info("✓ Quantization completed successfully!")
            logger.info("  (Use --validate to test the quantized model)")
            return 0

    except Exception as e:
        logger.error(f"✗ Quantization failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
