# Chronos: Learning the Language of Time Series

Chronos is a family of pre-trained **time series forecasting models** based on language model architectures. A time series is transformed into a sequence of tokens via scaling and quantization, and probabilistic forecasts are obtained by sampling multiple future trajectories given the historical context. Chronos models are trained on a large corpus of publicly available time series data, as well as synthetic data.

For details on Chronos models, training data and procedures, and experimental results, please refer to the paper [Chronos: Learning the Language of Time Series](https://www.example.com/).

## Architecture

The models in this repository are based on the [T5 architecture](https://arxiv.org/abs/1910.10683). The only difference is in the vocabulary size: Chronos-T5 models use 4096 different tokens, compared to 32128 of the original T5 models, resulting in a smaller number of parameters.

|Model	|Parameters	|Based on	|
|---	|---	|---	|
|[chronos-t5-tiny](https://huggingface.co/amazon/chronos-t5-tiny)	|8M	|[t5-efficient-tiny](https://huggingface.co/google/t5-efficient-tiny)	|
|[chronos-t5-mini](https://huggingface.co/amazon/chronos-t5-mini)	|20M	|[t5-efficient-mini](https://huggingface.co/google/t5-efficient-mini)	|
|[chronos-t5-small](https://huggingface.co/amazon/chronos-t5-small)	|46M	|[t5-efficient-small](https://huggingface.co/google/t5-efficient-small)	|
|[chronos-t5-base](https://huggingface.co/amazon/chronos-t5-base)	|200M	|[t5-efficient-base](https://huggingface.co/google/t5-efficient-base)	|
|[chronos-t5-large](https://huggingface.co/amazon/chronos-t5-large)	|710M	|[t5-efficient-large](https://huggingface.co/google/t5-efficient-large)	|

## Usage

To perform inference with Chronos models, you need to install the [companion inference package available on Github](https://www.example.com/).

```
pip install git+https://github.com/amazon-science/chronos-forecasting.git
```

A minimal example:

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline

pipeline = ChronosPipeline.from_pretrained("amazon/chronos-t5-small")

df = pd.read_csv("https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv")

# context must be either a 1D tensor, a list of 1D tensors, 
# or a left-padded 2D tensor with batch as the first dimension
context = torch.tensor(df["#Passengers"])
prediction_length = 12
forecast = pipeline.predict(context, prediction_length)  # shape [num_series, num_samples, prediction_length]

# visualize the forecast
forecast_index = range(len(df), len(df) + prediction_length)
low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

plt.plot(df["#Passengers"], color="royalblue", label="historical data")
plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")
plt.plot(forecast_index, median, color="tomato", label="median forecast")
plt.legend()
plt.grid()
plt.show()
```

## Citation

If you find Chronos models useful for your research, please consider citing the associated [paper](https://www.example.com/):

```
@article{ansari2024chronos,
  author  = {Ansari, Abdul Fatir and Stella, Lorenzo and Turkmen, Caner and Zhang, Xiyuan, and Mercado, Pedro and Shen, Huibin and Shchur, Oleksandr and Rangapuram, Syama Syndar and Pineda Arango, Sebastian and Kapoor, Shubham and Zschiegner, Jasper and Maddix, Danielle C. and Mahoney, Michael W. and Torkkola, Kari and Gordon Wilson, Andrew and Bohlke-Schneider, Michael and Wang, Yuyang},
  title   = {Chronos: Learning the Language of Time Series},
  journal = {arXiv preprint arXiv:ABCD.EFGH},
  year    = {2024}
}
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
