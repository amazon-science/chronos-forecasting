<div align="center">
<img src="https://raw.githubusercontent.com/amazon-science/chronos-forecasting/main/figures/chronos-logo.png" width="60%">
</div>

<div align="center">

# Chronos: Pretrained Models for Forecasting

[![preprint](https://img.shields.io/static/v1?label=arXiv&message=2403.07815&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2403.07815)
[![preprint](https://img.shields.io/static/v1?label=arXiv&message=2510.15821&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2510.15821)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Datasets-FFD21E)](https://huggingface.co/datasets/autogluon/chronos_datasets)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Models-FFD21E)](https://huggingface.co/collections/amazon/chronos-models-65f1791d630a8d57cb718444)
[![fev](https://img.shields.io/static/v1?label=fev&message=Benchmark&color=B31B1B&logo=github)](https://github.com/autogluon/fev)
[![aws](https://img.shields.io/static/v1?label=SageMaker&message=Deploy&color=FF9900&logo=amazon-web-services)](notebooks/deploy-chronos-bolt-to-amazon-sagemaker.ipynb)
[![faq](https://img.shields.io/badge/FAQ-Questions%3F-blue)](https://github.com/amazon-science/chronos-forecasting/issues?q=is%3Aissue+label%3AFAQ)
[![License: MIT](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

</div>


## 🚀 News
- **20 Oct 2025**: 🚀 [Chronos-2](https://arxiv.org/abs/2510.15821) released. It offers _zero-shot_ support for univariate, multivariate, and covariate-informed forecasting tasks. Chronos-2 achieves the best performance on fev-bench, GIFT-Eval and Chronos Benchmark II amongst pretrained models. Check out [this notebook]() to get started with Chronos-2.
- **14 Feb 2025**: 🚀 Chronos-Bolt is now available on Amazon SageMaker JumpStart! Check out the [tutorial notebook](notebooks/deploy-chronos-bolt-to-amazon-sagemaker.ipynb) to learn how to deploy Chronos endpoints for production use in 3 lines of code.
- **12 Dec 2024**: 📊 We released [`fev`](https://github.com/autogluon/fev), a lightweight package for benchmarking time series forecasting models based on the [Hugging Face `datasets`](https://huggingface.co/docs/datasets/en/index) library.
- **26 Nov 2024**: ⚡️ Chronos-Bolt models released [on HuggingFace](https://huggingface.co/collections/amazon/chronos-models-65f1791d630a8d57cb718444). Chronos-Bolt models are more accurate (5% lower error), up to 250x faster and 20x more memory efficient than the original Chronos models of the same size!
- **13 Mar 2024**: 🚀 Chronos [paper](https://arxiv.org/abs/2403.07815) and inference code released.

## ✨ Introduction

Chronos is a family of **pretrained time series forecasting models** based on language model architectures. A time series is transformed into a sequence of tokens via scaling and quantization, and a language model is trained on these tokens using the cross-entropy loss. Once trained, probabilistic forecasts are obtained by sampling multiple future trajectories given the historical context. Chronos models have been trained on a large corpus of publicly available time series data, as well as synthetic data generated using Gaussian processes.

For details on Chronos models, training data and procedures, and experimental results, please refer to the paper [Chronos: Learning the Language of Time Series](https://arxiv.org/abs/2403.07815).

<p align="center">
  <img src="https://raw.githubusercontent.com/amazon-science/chronos-forecasting/main/figures/main-figure.png" width="100%">
  <br />
  <span>
    Fig. 1: High-level depiction of Chronos. (<b>Left</b>) The input time series is scaled and quantized to obtain a sequence of tokens. (<b>Center</b>) The tokens are fed into a language model which may either be an encoder-decoder or a decoder-only model. The model is trained using the cross-entropy loss. (<b>Right</b>) During inference, we autoregressively sample tokens from the model and map them back to numerical values. Multiple trajectories are sampled to obtain a predictive distribution.
  </span>
</p>

### Architecture

The models in this repository are based on the [T5 architecture](https://arxiv.org/abs/1910.10683). The only difference is in the vocabulary size: Chronos-T5 models use 4096 different tokens, compared to 32128 of the original T5 models, resulting in fewer parameters.

<div align="center">

| Model                                                                  | Parameters | Based on                                                               |
| ---------------------------------------------------------------------- | ---------- | ---------------------------------------------------------------------- |
| [**chronos-t5-tiny**](https://huggingface.co/amazon/chronos-t5-tiny)   | 8M         | [t5-efficient-tiny](https://huggingface.co/google/t5-efficient-tiny)   |
| [**chronos-t5-mini**](https://huggingface.co/amazon/chronos-t5-mini)   | 20M        | [t5-efficient-mini](https://huggingface.co/google/t5-efficient-mini)   |
| [**chronos-t5-small**](https://huggingface.co/amazon/chronos-t5-small) | 46M        | [t5-efficient-small](https://huggingface.co/google/t5-efficient-small) |
| [**chronos-t5-base**](https://huggingface.co/amazon/chronos-t5-base)   | 200M       | [t5-efficient-base](https://huggingface.co/google/t5-efficient-base)   |
| [**chronos-t5-large**](https://huggingface.co/amazon/chronos-t5-large) | 710M       | [t5-efficient-large](https://huggingface.co/google/t5-efficient-large) |
| [**chronos-bolt-tiny**](https://huggingface.co/amazon/chronos-bolt-tiny)   | 9M         | [t5-efficient-tiny](https://huggingface.co/google/t5-efficient-tiny)   |
| [**chronos-bolt-mini**](https://huggingface.co/amazon/chronos-bolt-mini)   | 21M        | [t5-efficient-mini](https://huggingface.co/google/t5-efficient-mini)   |
| [**chronos-bolt-small**](https://huggingface.co/amazon/chronos-bolt-small) | 48M        | [t5-efficient-small](https://huggingface.co/google/t5-efficient-small) |
| [**chronos-bolt-base**](https://huggingface.co/amazon/chronos-bolt-base)   | 205M       | [t5-efficient-base](https://huggingface.co/google/t5-efficient-base)   |

</div>

### Zero-Shot Results

The following figure showcases the remarkable **zero-shot** performance of Chronos and Chronos-Bolt models on 27 datasets against local models, task-specific models and other pretrained models. For details on the evaluation setup and other results, please refer to [the paper](https://arxiv.org/abs/2403.07815).

<p align="center">
  <img src="https://raw.githubusercontent.com/amazon-science/chronos-forecasting/main/figures/zero_shot-agg_scaled_score.svg" width="100%">
  <br />
  <span>
    Fig. 2: Performance of different models on Benchmark II, comprising 27 datasets <b>not seen</b> by Chronos and Chronos-Bolt models during training. This benchmark provides insights into the zero-shot performance of Chronos and Chronos-Bolt models against local statistical models, which fit parameters individually for each time series, task-specific models <i>trained on each task</i>, and pretrained models trained on a large corpus of time series. Pretrained Models (Other) indicates that some (or all) of the datasets in Benchmark II may have been in the training corpus of these models. The probabilistic (WQL) and point (MASE) forecasting metrics were normalized using the scores of the Seasonal Naive baseline and aggregated through a geometric mean to obtain the Agg. Relative WQL and MASE, respectively.
  </span>
</p>

## 📈 Usage

To perform inference with Chronos or Chronos-Bolt models, the easiest way is to install this package through `pip`:

```sh
pip install chronos-forecasting
```

If you're interested in pretraining, fine-tuning, and other research & development, clone and install the package from source:

```sh
# Clone the repository
git clone https://github.com/amazon-science/chronos-forecasting.git

# Install in editable mode with extra training-related dependencies
cd chronos-forecasting && pip install --editable ".[training]"
```

> [!TIP]
> This repository is intended for research purposes and provides a minimal interface to Chronos models. For reliable production use, we recommend the following options:
> - [AutoGluon](https://auto.gluon.ai) provides effortless fine-tuning, augmenting Chronos models with exogenous information through covariate regressors, ensembling with other statistical and machine learning models. Check out the AutoGluon Chronos [tutorial](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-chronos.html).
> - SageMaker JumpStart makes it easy to deploy Chronos inference endpoints to AWS with just a few lines of code. Check out [this tutorial](notebooks/deploy-chronos-bolt-to-amazon-sagemaker.ipynb) for more details.

### Forecasting

A minimal example showing how to perform forecasting using Chronos and Chronos-Bolt models:

```python
import pandas as pd  # requires: pip install pandas
import torch
from chronos import BaseChronosPipeline

pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",  # use "amazon/chronos-bolt-small" for the corresponding Chronos-Bolt model
    device_map="cuda",  # use "cpu" for CPU inference
    torch_dtype=torch.bfloat16,
)

df = pd.read_csv(
    "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv"
)

# context must be either a 1D tensor, a list of 1D tensors,
# or a left-padded 2D tensor with batch as the first dimension
# quantiles is an fp32 tensor with shape [batch_size, prediction_length, num_quantile_levels]
# mean is an fp32 tensor with shape [batch_size, prediction_length]
quantiles, mean = pipeline.predict_quantiles(
    context=torch.tensor(df["#Passengers"]),
    prediction_length=12,
    quantile_levels=[0.1, 0.5, 0.9],
)
```

For the original Chronos models, `pipeline.predict` can be used to draw forecast samples. More options for `predict_kwargs` in `pipeline.predict_quantiles` can be found with:

```python
from chronos import ChronosPipeline, ChronosBoltPipeline

print(ChronosPipeline.predict.__doc__)  # for Chronos models
print(ChronosBoltPipeline.predict.__doc__)  # for Chronos-Bolt models
```

We can now visualize the forecast:

```python
import matplotlib.pyplot as plt  # requires: pip install matplotlib

forecast_index = range(len(df), len(df) + 12)
low, median, high = quantiles[0, :, 0], quantiles[0, :, 1], quantiles[0, :, 2]

plt.figure(figsize=(8, 4))
plt.plot(df["#Passengers"], color="royalblue", label="historical data")
plt.plot(forecast_index, median, color="tomato", label="median forecast")
plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")
plt.legend()
plt.grid()
plt.show()
```

### Extracting Encoder Embeddings

A minimal example showing how to extract encoder embeddings from Chronos models:

```python
import pandas as pd
import torch
from chronos import ChronosPipeline

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)

df = pd.read_csv("https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv")

# context must be either a 1D tensor, a list of 1D tensors,
# or a left-padded 2D tensor with batch as the first dimension
context = torch.tensor(df["#Passengers"])
embeddings, tokenizer_state = pipeline.embed(context)
```

### Pretraining, fine-tuning and evaluation

Scripts for pretraining, fine-tuning and evaluating Chronos models can be found in [this folder](./scripts/).

## :floppy_disk: Datasets

Datasets used in the Chronos paper for pretraining and evaluation (both in-domain and zero-shot) are available through the HuggingFace repos: [`autogluon/chronos_datasets`](https://huggingface.co/datasets/autogluon/chronos_datasets) and [`autogluon/chronos_datasets_extra`](https://huggingface.co/datasets/autogluon/chronos_datasets_extra). Check out these repos for instructions on how to download and use the datasets.

## 📝 Citation

If you find this  models useful for your research, please consider citing the associated papers:

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
