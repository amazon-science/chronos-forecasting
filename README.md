# ChronosX: Adapting Pretrained Time Series Models with Exogenous Variables

[![preprint](https://img.shields.io/static/v1?label=arXiv&message=2503.12107&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2503.12107)
[![License: MIT](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

This repository provides the forecasting model ChronosX introduced in the paper
[ChronosX: Adapting Pretrained Time Series Models with Exogenous Variables](https://arxiv.org/abs/2503.12107).

## ✨ Introduction
ChronosX introduces a new method to incorporate covariates into pretrained time series forecasting models. ChronosX incorporates covariate information into pretrained forecasting models through modular blocks that inject past and future covariate information, without necessarily modifying the pretrained model in consideration.

For further details on ChronosX please refer to the paper [ChronosX: Adapting Pretrained Time Series Models with Exogenous Variables](https://arxiv.org/abs/2503.12107).

## 📈 Usage

To perform inference with Chronos or Chronos-Bolt models, the easiest way is to install this package through `pip`:

```sh
pip install git+https://github.com/amazon-science/chronos-forecasting.git@chronosx
```

### Forecasting

A minimal example showing how to perform forecasting using ChronosX:

```python
import numpy as np
import yaml

from chronosx.chronosx import ChronosXPipeline
from chronosx.utils import ChronosDataset, load_and_split_dataset
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
from gluonts.model.evaluation import evaluate_forecasts
from pathlib import Path


config_path = "./scripts/experiments/configs/datasets.yaml"
output_dir = Path(f"./output/finetune")
output_dir.mkdir(exist_ok=True, parents=True)

with open(config_path) as fp:
    backtest_configs = yaml.safe_load(fp)

dataset_config = backtest_configs[0]
prediction_length = dataset_config["prediction_length"]
num_covariates = 2 * len(dataset_config["covariates_fields"])


# Load Chronos
pipeline = ChronosXPipeline(
    pretrained_model_name_or_path="amazon/chronos-t5-small",
    prediction_length=prediction_length,
    num_covariates=num_covariates,
)

# load Dataset
train_dataset, test_dataset = load_and_split_dataset(backtest_config=dataset_config)
quantized_train_dataset = ChronosDataset(
    datasets=[train_dataset],
    probabilities=[1.0],
    tokenizer=pipeline.tokenizer,
    prediction_length=prediction_length,
    mode="training",
).shuffle()

# fine tune model
_, save_model_path = pipeline.finetune(
    output_dir,
    quantized_train_dataset,
    skip_pretrained_validation=True,
)

# Evaluate fine tuned model
pipeline = ChronosXPipeline(
    prediction_length=prediction_length,
    num_covariates=num_covariates,
    pretrained_model_name_or_path=output_dir / "final-checkpoint",
)

pipeline.chronosx.eval()
forecasts = pipeline.generate_forecasts(test_dataset.input)

metrics = (
    evaluate_forecasts(
        forecasts,
        test_data=test_dataset,
        metrics=[
            MASE(),
            MeanWeightedSumQuantileLoss(np.arange(0.05, 1, 0.05).round(2).tolist()),
        ],
    )
    .reset_index(drop=True)
    .to_dict(orient="records")
)

print(metrics)
```

### Experiments

Scripts for experiments can be found in [this folder](./scripts/).

## 📝 Citation

If you find ChronosX useful for your research, please consider citing the associated [paper](https://arxiv.org/abs/2503.12107):

```
@inproceedings{
arango2025chronosx,
title={ChronosX: Adapting Pretrained Time Series Models with Exogenous Variables},
author={Sebastian Pineda Arango and Pedro Mercado and Shubham Kapoor and Abdul Fatir Ansari and Lorenzo Stella and Huibin Shen and Hugo Henri Joseph Senetaire and Ali Caner Turkmen and Oleksandr Shchur and Danielle C. Maddix and Bernie Wang and Michael Bohlke-Schneider and Syama Sundar Rangapuram},
booktitle={The 28th International Conference on Artificial Intelligence and Statistics},
year={2025},
url={https://openreview.net/forum?id=f4nWNn0RjV}
}
```

## 🛡️ Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## 📃 License

This project is licensed under the Apache-2.0 License.