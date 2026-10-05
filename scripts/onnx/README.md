# Chronos-2 ONNX Export

These scripts export the Chronos-2 tensor model to ONNX, repair the exported graph, and validate ONNX Runtime output against the PyTorch model.

The exporter writes a real ONNX model. It does not commit or vendor generated model artifacts into this repository.

## Install

Install Chronos and the ONNX dependencies in an environment with a current PyTorch release:

```bash
pip install torch onnx onnxruntime onnxscript transformers chronos-forecasting
```

Use `onnxruntime-gpu` instead of `onnxruntime` if you want CUDA inference.

## Export

Export the public Chronos-2 model with future covariates enabled:

```bash
python scripts/onnx/export_chronos2_to_onnx.py \
  --model_id amazon/chronos-2 \
  --output_dir chronos2-onnx \
  --validate
```

The exporter first writes `model_raw.onnx`, then runs `fix_onnx_model.py` and writes the final loadable model to `model.onnx`. The raw model is deleted unless `--keep_raw_onnx` is passed.

Important options:

- `--context_length`: fixed context length to trace into the ONNX graph. Default: `512`.
- `--num_output_patches`: fixed number of output patches to trace. Default: `4`.
- `--no_future_covariates`: export without the `future_covariates` input.
- `--no_fix_onnx`: skip the graph repair pass. This is useful only for debugging; the raw graph may not load in ONNX Runtime.
- `--quantize`: additionally write a dynamic INT8 quantized model.

For the default Chronos-2 config, `output_patch_size=16`, so `--num_output_patches 4` exports a 64-step horizon.

## Validate Parity

The export script can run a basic PyTorch-vs-ONNX validation with `--validate`. For fuller coverage, run the standalone parity harness:

```bash
python scripts/onnx/validate_chronos2_onnx.py \
  --model_id amazon/chronos-2 \
  --onnx_path chronos2-onnx/model.onnx \
  --context_length 512 \
  --num_output_patches 4 \
  --report_path chronos2-onnx/parity_report.json
```

The harness compares the ONNX output with the PyTorch wrapper across several cases:

- dynamic batch sizes
- shared and distinct `group_ids`
- sinusoidal, random, and zero future covariates
- missing context values
- missing future covariate values

It exits nonzero if any case fails `np.allclose`.

## Tensor Interface

The exported model exposes the Chronos-2 tensor interface used by `Chronos2Model.forward`.

Inputs:

- `context`: float32 tensor shaped `[batch, context_length]`.
- `group_ids`: int64 tensor shaped `[batch]`. Series with equal IDs form an attention group.
- `attention_mask`: float32 tensor shaped `[batch, context_length]`, where `1` marks observed positions and `0` marks masked positions.
- `future_covariates`: optional float32 tensor shaped `[batch, prediction_length]`, present unless `--no_future_covariates` is used.
- `num_output_patches`: optional int64 scalar. Some PyTorch legacy exports expose this scalar input. If present, feed the same value used during export.

Output:

- `quantile_preds`: float32 tensor shaped `[batch, num_quantiles, prediction_length]`.

`prediction_length = num_output_patches * output_patch_size`. The default export for `amazon/chronos-2` is `[batch, 21, 64]`.

Minimal ONNX Runtime call:

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("chronos2-onnx/model.onnx", providers=["CPUExecutionProvider"])
input_names = {input_.name for input_ in session.get_inputs()}

batch_size = 2
context_length = 512
num_output_patches = 4
prediction_length = 64

inputs = {
    "context": np.random.randn(batch_size, context_length).astype(np.float32),
    "group_ids": np.arange(batch_size, dtype=np.int64),
    "attention_mask": np.ones((batch_size, context_length), dtype=np.float32),
}

if "future_covariates" in input_names:
    inputs["future_covariates"] = np.random.randn(batch_size, prediction_length).astype(np.float32)

if "num_output_patches" in input_names:
    inputs["num_output_patches"] = np.array(num_output_patches, dtype=np.int64)

quantile_preds = session.run(None, inputs)[0]
```

## Repairing a Raw Export

`fix_onnx_model.py` repairs Gather index dtype mismatches emitted by the legacy PyTorch exporter:

```bash
python scripts/onnx/fix_onnx_model.py model_raw.onnx model.onnx
```

The fixer does not mark prediction length dynamic by default, because the traced ONNX graph has a fixed executable horizon. For covariate exports it infers the fixed output length from `future_covariates`; for non-covariate exports you can pass `--prediction_length`. `--dynamic_prediction_length` only changes output shape metadata and should not be treated as runtime support for arbitrary horizons.

## Supported Shapes and Limitations

- Batch size is dynamic.
- Context length is fixed at export time.
- Future covariate length is fixed at export time and should match the exported prediction length.
- Prediction length is fixed at export time.
- The tensor-level export does not include the `Chronos2Pipeline.predict` list-of-dicts/DataFrame preprocessing wrapper. Prepare `context`, `group_ids`, `attention_mask`, and optional `future_covariates` tensors before calling ONNX Runtime.
- Missing future covariate values can be represented as `NaN`; Chronos-2 infers the future covariate mask from those values when no explicit mask is exported.
- Quantized models should be validated separately. Dynamic quantization can change numeric parity.
- Browser, server, and application packaging are intentionally outside this export contract.
