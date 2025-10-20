<div align="center">
<img src="https://raw.githubusercontent.com/amazon-science/chronos-forecasting/main/figures/chronos-logo.png" width="60%">
</div>

<div align="center">

# Chronos: Pretrained Models for Time Series Forecasting

[![preprint](https://img.shields.io/static/v1?label=Chronos-Paper&message=2403.07815&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2403.07815)
[![preprint](https://img.shields.io/static/v1?label=Chronos-2-Report&message=2510.15821&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2510.15821)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Datasets-FFD21E)](https://huggingface.co/datasets/autogluon/chronos_datasets)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Models-FFD21E)](https://huggingface.co/collections/amazon/chronos-models-65f1791d630a8d57cb718444)
[![fev](https://img.shields.io/static/v1?label=fev&message=Benchmark&color=B31B1B&logo=github)](https://github.com/autogluon/fev)
[![aws](https://img.shields.io/static/v1?label=SageMaker&message=Deploy&color=FF9900&logo=amazon-web-services)](notebooks/deploy-chronos-bolt-to-amazon-sagemaker.ipynb)
[![faq](https://img.shields.io/badge/FAQ-Questions%3F-blue)](https://github.com/amazon-science/chronos-forecasting/issues?q=is%3Aissue+label%3AFAQ)
[![License: MIT](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

</div>


## 🚀 News
- **20 Oct 2025**: 🚀 [Chronos-2](https://arxiv.org/abs/2510.15821) released. It offers _zero-shot_ support for univariate, multivariate, and covariate-informed forecasting tasks. Chronos-2 achieves the best performance on fev-bench, GIFT-Eval and Chronos Benchmark II amongst pretrained models. Check out [this notebook](notebooks/chronos-2-quickstart.ipynb) to get started with Chronos-2.
- **14 Feb 2025**: 🚀 Chronos-Bolt is now available on Amazon SageMaker JumpStart! Check out the [tutorial notebook](notebooks/deploy-chronos-bolt-to-amazon-sagemaker.ipynb) to learn how to deploy Chronos endpoints for production use in 3 lines of code.
- **12 Dec 2024**: 📊 We released [`fev`](https://github.com/autogluon/fev), a lightweight package for benchmarking time series forecasting models based on the [Hugging Face `datasets`](https://huggingface.co/docs/datasets/en/index) library.
- **26 Nov 2024**: ⚡️ Chronos-Bolt models released [on HuggingFace](https://huggingface.co/collections/amazon/chronos-models-65f1791d630a8d57cb718444). Chronos-Bolt models are more accurate (5% lower error), up to 250x faster and 20x more memory efficient than the original Chronos models of the same size!
- **13 Mar 2024**: 🚀 Chronos [paper](https://arxiv.org/abs/2403.07815) and inference code released.

## ✨ Introduction

This package provides an interface to the Chronos family of **pretrained time series forecasting models** developed by AWS. The following model families are supported.

- **Chronos**: The original Chronos family which is based on language model architectures. A time series is transformed into a sequence of tokens via scaling and quantization, and a language model is trained on these tokens using the cross-entropy loss. Once trained, probabilistic forecasts are obtained by sampling multiple future trajectories given the historical context. To learn more about Chronos, check out the [publication](https://openreview.net/forum?id=gerNCVqqtR).
- **Chronos-Bolt**: A patch-based variant of Chronos. It chunks the historical time series context into patches of multiple observations, which are then input into the encoder. The decoder then uses these representations to directly generate quantile forecasts across multiple future steps—a method known as direct multi-step forecasting. Chronos-Bolt models are up to 250 times faster and 20 times more memory-efficient than the original Chronos models of the same size. To learn more about Chronos-Bolt, check out this [blog post](https://aws.amazon.com/blogs/machine-learning/fast-and-accurate-zero-shot-forecasting-with-chronos-bolt-and-autogluon/).
- **Chronos-2**: Our latest model with significantly enhanced capabilities. It offers zero-shot support for univariate, multivariate, and covariate-informed forecasting tasks. Chronos-2 delivers state-of-the-art zero-shot performance across multiple benchmarks (including fev-bench and GIFT-Eval), with the largest improvements observed on tasks that include exogenous features. It also achieves a win rate of over 90% against Chronos-Bolt in head-to-head comparisons. To learn more about Chronos, check out the [technical report](https://arxiv.org/abs/2510.15821).

### Available Models

<div align="center">

| Model                                                                  | Parameters |
| ---------------------------------------------------------------------- | ---------- |
| `s3://autogluon/chronos-2`   | 120M         |
| [`amazon/chronos-bolt-tiny`](https://huggingface.co/amazon/chronos-bolt-tiny)   | 9M         |
| [`amazon/chronos-bolt-mini`](https://huggingface.co/amazon/chronos-bolt-mini)   | 21M        |
| [`amazon/chronos-bolt-small`](https://huggingface.co/amazon/chronos-bolt-small) | 48M        |
| [`amazon/chronos-bolt-base`](https://huggingface.co/amazon/chronos-bolt-base)   | 205M       |
| [`amazon/chronos-t5-tiny`](https://huggingface.co/amazon/chronos-t5-tiny)   | 8M         |
| [`amazon/chronos-t5-mini`](https://huggingface.co/amazon/chronos-t5-mini)   | 20M        |
| [`amazon/chronos-t5-small`](https://huggingface.co/amazon/chronos-t5-small) | 46M        |
| [`amazon/chronos-t5-base`](https://huggingface.co/amazon/chronos-t5-base)   | 200M       |
| [`amazon/chronos-t5-large`](https://huggingface.co/amazon/chronos-t5-large) | 710M       | 

</div>

## 📈 Usage

To perform inference with Chronos, the easiest way is to install this package through `pip`:

```sh
pip install chronos-forecasting
```

### Forecasting

A minimal example showing how to perform forecasting using Chronos-2:

```python
import pandas as pd  # requires: pip install pandas
from chronos import Chronos2Pipeline

pipeline = Chronos2Pipeline.from_pretrained("s3://autogluon/chronos-2", device_map="cuda")

target = "target"  # Column name containing the values to forecast
prediction_length = 24  # Number of steps to forecast ahead
id_column = "id"  # Column identifying different time series
timestamp_column = "timestamp"  # Column containing datetime information

# Load historical energy prices and past values of covariates
context_df = pd.read_parquet("https://autogluon.s3.amazonaws.com/datasets/timeseries/electricity_price/train.parquet")

# Load future values of covariates
test_df = pd.read_parquet("s3://autogluon/datasets/timeseries/electricity_price/test.parquet")
future_df = test_df.drop(columns=target)

# Generate predictions with covariates
pred_df = pipeline.predict_df(
    context_df,
    future_df=future_df,
    prediction_length=prediction_length,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column=id_column,
    timestamp_column=timestamp_column,
    target=target,
)
```

We can now visualize the forecast:

```python
import matplotlib.pyplot as plt  # requires: pip install matplotlib

timeseries_id = "DE"  # Specific time series to visualize
history_length = 256  # The number of historical values to plot

ts_context = context_df.query(f"{id_column} == @timeseries_id").set_index(timestamp_column)[target]
ts_pred = pred_df.query(f"{id_column} == @timeseries_id and target_name == @target").set_index(timestamp_column)[
    ["0.1", "predictions", "0.9"]
]
ts_ground_truth = test_df.query(f"{id_column} == @timeseries_id").set_index(timestamp_column)[target]

start_idx = max(0, len(ts_context) - history_length)
plot_cutoff = ts_context.index[start_idx]
ts_context = ts_context[ts_context.index >= plot_cutoff]
ts_ground_truth = ts_ground_truth[ts_ground_truth.index >= plot_cutoff]

fig = plt.figure(figsize=(12, 3))
ax = fig.gca()
ts_context.plot(ax=ax, label=f"historical {target}", color="xkcd:azure")
ts_ground_truth.plot(ax=ax, label=f"future {target} (ground truth)", color="xkcd:grass green")
ts_pred["predictions"].plot(ax=ax, label="forecast", color="xkcd:violet")
ax.fill_between(
    ts_pred.index,
    ts_pred["0.1"],
    ts_pred["0.9"],
    alpha=0.7,
    label="prediction interval",
    color="xkcd:light lavender",
)
ax.legend(loc="upper left")
ax.set_title(f"{target} forecast for {timeseries_id}")
fig.show()
```

## Example Notebooks

- [Chronos-2 Quick Start](notebooks/chronos-2-quickstart.ipynb)
- [Deploy Chronos-Bolt on Amazon SageMaker](notebooks/deploy-chronos-bolt-to-amazon-sagemaker.ipynb)
- Deploy Chronos-2 on Amazon SageMaker (coming soon!)

## 📝 Citation

If you find Chronos models useful for your research, please consider citing the associated papers:

```
@article{ansari2024chronos,
  title={Chronos: Learning the Language of Time Series},
  author={Ansari, Abdul Fatir and Stella, Lorenzo and Turkmen, Caner and Zhang, Xiyuan, and Mercado, Pedro and Shen, Huibin and Shchur, Oleksandr and Rangapuram, Syama Syndar and Pineda Arango, Sebastian and Kapoor, Shubham and Zschiegner, Jasper and Maddix, Danielle C. and Mahoney, Michael W. and Torkkola, Kari and Gordon Wilson, Andrew and Bohlke-Schneider, Michael and Wang, Yuyang},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2024},
  url={https://openreview.net/forum?id=gerNCVqqtR}
}

@article{ansari2025chronos2,
  title        = {Chronos-2: From Univariate to Universal Forecasting},
  author       = {Abdul Fatir Ansari and Oleksandr Shchur and Jaris Küken and Andreas Auer and Boran Han and Pedro Mercado and Syama Sundar Rangapuram and Huibin Shen and Lorenzo Stella and Xiyuan Zhang and Mononito Goswami and Shubham Kapoor and Danielle C. Maddix and Pablo Guerron and Tony Hu and Junming Yin and Nick Erickson and Prateek Mutalik Desai and Hao Wang and Huzefa Rangwala and George Karypis and Yuyang Wang and Michael Bohlke-Schneider},
  journal      = {arXiv preprint arXiv:2510.15821},
  year         = {2025},
  url          = {https://arxiv.org/abs/2510.15821}
}
```

## 🛡️ Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## 📃 License

This project is licensed under the Apache-2.0 License.
