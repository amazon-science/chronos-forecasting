from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from chronos import BaseChronosPipeline

AIR_PASSENGERS_CSV = "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv"


def forecast_and_plot(model_id: str = "amazon/chronos-bolt-tiny"):
    pipeline = BaseChronosPipeline.from_pretrained(
        model_id,
        device_map="cpu",
        torch_dtype=torch.bfloat16,
    )

    df = pd.read_csv(AIR_PASSENGERS_CSV)

    quantiles, _ = pipeline.predict_quantiles(
        context=torch.tensor(df["#Passengers"]),
        prediction_length=12,
        quantile_levels=[0.1, 0.5, 0.9],
    )

    forecast_index = range(len(df), len(df) + 12)
    low, median, high = quantiles[0, :, 0], quantiles[0, :, 1], quantiles[0, :, 2]

    plt.figure(figsize=(8, 4))
    plt.plot(df["#Passengers"], color="royalblue", label="historical data")
    plt.plot(forecast_index, median, color="tomato", label="median forecast")
    plt.fill_between(
        forecast_index,
        low,
        high,
        color="tomato",
        alpha=0.3,
        label="80% prediction interval",
    )
    plt.title(model_id)
    plt.legend()
    plt.grid()
    plt.savefig(
        Path(__file__).parent / f"{model_id.replace('/', '_')}-forecast.png"
    )


if __name__ == "__main__":
    forecast_and_plot(model_id="amazon/chronos-bolt-tiny")
