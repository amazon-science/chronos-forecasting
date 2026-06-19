# Chronos-2 vLLM Plugin

A [vLLM](https://github.com/vllm-project/vllm) plugin that adds support for [Chronos-2](https://github.com/amazon-science/chronos-forecasting) time series forecasting via the `/pooling` API endpoint.

## Overview

Chronos-2 is an encoder-only time series foundation model for zero-shot forecasting. This plugin integrates it with vLLM using the **IOProcessor** plugin interface, so forecast requests are served through vLLM's standard pooling endpoint.

### Features

- **Zero-shot forecasting** — no fine-tuning required
- **Quantile predictions** — probabilistic forecasts with customizable quantile levels
- **Cross-series learning** — information sharing across time series in a batch
- **Covariates support** — past and future covariates (numeric and categorical)
- **Batch forecasting** — process multiple time series in a single request

## Installation

Requires Python 3.10+ and vLLM 0.13.0+.

```bash
pip install chronos-forecasting[vllm]
```

## Quick Start

### 1. Start the Server

```bash
vllm serve amazon/chronos-2 \
    --io-processor-plugin chronos2 \
    --runner pooling \
    --enforce-eager \
    --no-enable-prefix-caching \
    --skip-tokenizer-init \
    --enable-mm-embeds \
    --dtype float32 \
    --max-model-len 8192
```

### 2. Send a Forecast Request

```bash
curl -X POST http://localhost:8000/pooling \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon/chronos-2",
    "task": "plugin",
    "data": {
      "inputs": [
        {
          "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
          "item_id": "series_1"
        }
      ],
      "parameters": {
        "prediction_length": 5,
        "quantile_levels": [0.1, 0.5, 0.9]
      }
    }
  }'
```

### 3. Parse the Response

```json
{
  "request_id": null,
  "created_at": 1739397600,
  "data": {
    "predictions": [
      {
        "mean": [11.0, 12.1, 13.0, 14.2, 15.1],
        "0.1": [9.5, 10.3, 11.0, 11.8, 12.5],
        "0.5": [11.0, 12.0, 13.0, 14.0, 15.0],
        "0.9": [12.5, 13.8, 15.0, 16.3, 17.5],
        "item_id": "series_1"
      }
    ]
  }
}
```

## API Reference

### Request Format

| Field | Type | Required | Description |
|---|---|---|---|
| `model` | `str` | ✅ | Model name (e.g., `"amazon/chronos-2"`) |
| `task` | `str` | ✅ | Must be `"plugin"` |
| `data.inputs` | `list` | ✅ | List of time series inputs (1–1024) |
| `data.parameters` | `dict` | | Forecast parameters |

#### Time Series Input (`data.inputs[*]`)

| Field | Type | Required | Description |
|---|---|---|---|
| `target` | `list[float]` or `list[list[float]]` | ✅ | Historical values (min 5 observations). 1-D for univariate, 2-D for multivariate. |
| `item_id` | `str` | | Identifier echoed in response |
| `start` | `str` | | ISO 8601 timestamp |
| `past_covariates` | `dict[str, list]` | | Past covariate arrays (must match target length) |
| `future_covariates` | `dict[str, list]` | | Future covariate arrays (must match `prediction_length`) |

#### Parameters (`data.parameters`)

| Field | Type | Default | Description |
|---|---|---|---|
| `prediction_length` | `int` | `1` | Forecast horizon (1–1024) |
| `quantile_levels` | `list[float]` | `[0.1, 0.5, 0.9]` | Quantile levels in (0, 1) |
| `freq` | `str` | `null` | Pandas frequency string (e.g., `"D"`, `"H"`) |
| `batch_size` | `int` | `256` | Inference batch size |
| `cross_learning` | `bool` | `false` | Enable cross-series learning |

### Response Format

Each prediction in `data.predictions` contains:

| Field | Type | Description |
|---|---|---|
| `mean` | `list[float]` | Point forecast (mean/median) |
| `"0.1"`, `"0.5"`, etc. | `list[float]` | Named quantile columns matching `quantile_levels` |
| `item_id` | `str` | Echoed from input (if provided) |

## Architecture

The vLLM model wrapper (`Chronos2ForForecasting`) is a thin adapter that delegates all computation to the existing `chronos.chronos2.model.Chronos2Model`. No model architecture is duplicated.

### Module Structure

```
src/chronos/chronos2/vllm/
├── __init__.py          # Plugin entry point & registration
├── model.py             # Chronos2ForForecasting (thin vLLM wrapper)
├── multimodal.py        # MM pipeline for "timeseries" modality
├── io_processor.py      # Chronos2IOProcessor (request/response handling)
├── protocol/
│   ├── __init__.py
│   ├── forecast.py      # Pydantic models (TimeSeriesInput, ForecastParameters, etc.)
│   ├── validation.py    # Input validation logic
│   └── data_prep.py     # Tensor preparation from validated inputs
└── utils/
    ├── __init__.py
    ├── helpers.py        # Utility functions
    └── quantiles.py      # Quantile selection & interpolation
```

### Key Classes

| Class | File | Purpose |
|---|---|---|
| `Chronos2ForForecasting` | `model.py` | Thin vLLM wrapper — delegates to `chronos.chronos2.model.Chronos2Model` |
| `Chronos2IOProcessor` | `io_processor.py` | Request parsing, validation, pre/post processing |
| `ForecastParameters` | `protocol/forecast.py` | Pydantic validation for forecast parameters |
| `TimeSeriesInput` | `protocol/forecast.py` | Pydantic validation for time series inputs |
| `ForecastPrediction` | `protocol/forecast.py` | Pydantic model for forecast output |

## Troubleshooting

### Server Flags

The following flags are required for Chronos-2:

```bash
--io-processor-plugin chronos2   # Enable the forecast IOProcessor
--enforce-eager                   # Chronos-2 doesn't support CUDA graphs
--no-enable-prefix-caching        # Not applicable for time series
--skip-tokenizer-init             # Chronos-2 doesn't use a text tokenizer
```

### Plugin Not Loading

1. Verify installation: `pip list | grep chronos-forecasting`
2. Check entry points:
   ```python
   from importlib.metadata import entry_points
   print(list(entry_points(group='vllm.general_plugins')))
   ```
3. Enable debug logging: `VLLM_LOGGING_LEVEL=DEBUG vllm serve ...`