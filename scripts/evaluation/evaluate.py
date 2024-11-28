import logging
from pathlib import Path
from typing import Iterable, Optional

import datasets
import numpy as np
import pandas as pd
import torch
import typer
import yaml
from gluonts.dataset.split import split
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
from gluonts.itertools import batcher
from gluonts.model.evaluation import evaluate_forecasts
from gluonts.model.forecast import QuantileForecast, SampleForecast
from tqdm.auto import tqdm

from chronos import (
    BaseChronosPipeline,
    ChronosBoltPipeline,
    ChronosPipeline,
    ForecastType,
)

app = typer.Typer(pretty_exceptions_enable=False)


def to_gluonts_univariate(hf_dataset: datasets.Dataset):
    series_fields = [
        col
        for col in hf_dataset.features
        if isinstance(hf_dataset.features[col], datasets.Sequence)
    ]
    series_fields.remove("timestamp")
    dataset_length = hf_dataset.info.splits["train"].num_examples * len(series_fields)

    # Assumes that all time series in the dataset have the same frequency
    dataset_freq = pd.DatetimeIndex(hf_dataset[0]["timestamp"]).to_period()[0].freqstr

    gts_dataset = []
    for hf_entry in hf_dataset:
        for field in series_fields:
            gts_dataset.append(
                {
                    "start": pd.Period(
                        hf_entry["timestamp"][0],
                        freq=dataset_freq,
                    ),
                    "target": hf_entry[field],
                }
            )
    assert len(gts_dataset) == dataset_length

    return gts_dataset


def load_and_split_dataset(backtest_config: dict):
    hf_repo = backtest_config["hf_repo"]
    dataset_name = backtest_config["name"]
    offset = backtest_config["offset"]
    prediction_length = backtest_config["prediction_length"]
    num_rolls = backtest_config["num_rolls"]

    # This is needed because the datasets in autogluon/chronos_datasets_extra cannot
    # be distribued due to license restrictions and must be generated on the fly
    trust_remote_code = True if hf_repo == "autogluon/chronos_datasets_extra" else False

    ds = datasets.load_dataset(
        hf_repo, dataset_name, split="train", trust_remote_code=trust_remote_code
    )
    ds.set_format("numpy")

    gts_dataset = to_gluonts_univariate(ds)

    # Split dataset for evaluation
    _, test_template = split(gts_dataset, offset=offset)
    test_data = test_template.generate_instances(prediction_length, windows=num_rolls)

    return test_data


def generate_forecasts(
    test_data_input: Iterable,
    pipeline: BaseChronosPipeline,
    prediction_length: int,
    batch_size: int,
    **predict_kwargs,
):
    # Generate forecasts
    forecast_outputs = []
    for batch in tqdm(batcher(test_data_input, batch_size=batch_size)):
        context = [torch.tensor(entry["target"]) for entry in batch]
        forecast_outputs.append(
            pipeline.predict(
                context,
                prediction_length=prediction_length,
                **predict_kwargs,
            ).numpy()
        )
    forecast_outputs = np.concatenate(forecast_outputs)

    # Convert forecast samples into gluonts Forecast objects
    forecasts = []
    for item, ts in zip(forecast_outputs, test_data_input):
        forecast_start_date = ts["start"] + len(ts["target"])

        if pipeline.forecast_type == ForecastType.SAMPLES:
            forecasts.append(
                SampleForecast(samples=item, start_date=forecast_start_date)
            )
        elif pipeline.forecast_type == ForecastType.QUANTILES:
            forecasts.append(
                QuantileForecast(
                    forecast_arrays=item,
                    forecast_keys=list(map(str, pipeline.quantiles)),
                    start_date=forecast_start_date,
                )
            )

    return forecasts


@app.command()
def main(
    config_path: Path,
    metrics_path: Path,
    chronos_model_id: str = "amazon/chronos-t5-small",
    device: str = "cuda",
    torch_dtype: str = "bfloat16",
    batch_size: int = 32,
    num_samples: int = 20,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
):
    """Evaluate Chronos models.

    Parameters
    ----------
    config_path : Path
        Path to the evaluation config. See ./configs/.
    metrics_path : Path
        Path to the CSV file where metrics will be saved.
    chronos_model_id : str, optional, default = "amazon/chronos-t5-small"
        HuggingFace ID of the Chronos model or local path
        Available models on HuggingFace:
        Chronos:
            - amazon/chronos-t5-tiny
            - amazon/chronos-t5-mini
            - amazon/chronos-t5-small
            - amazon/chronos-t5-base
            - amazon/chronos-t5-large
        Chronos-Bolt:
            - amazon/chronos-bolt-tiny
            - amazon/chronos-bolt-mini
            - amazon/chronos-bolt-small
            - amazon/chronos-bolt-base
    device : str, optional, default = "cuda"
        Device on which inference will be performed
    torch_dtype : str, optional
        Model's dtype, by default "bfloat16"
    batch_size : int, optional, default = 32
        Batch size for inference. For Chronos-Bolt models, significantly larger
        batch sizes can be used
    num_samples : int, optional, default = 20
        Number of samples to draw when using the original Chronos models
    temperature : Optional[float], optional, default = 1.0
        Softmax temperature to used for the original Chronos models
    top_k : Optional[int], optional, default = 50
        Top-K sampling, by default None
    top_p : Optional[float], optional, default = 1.0
        Top-p sampling, by default None
    """
    if isinstance(torch_dtype, str):
        torch_dtype = getattr(torch, torch_dtype)
    assert isinstance(torch_dtype, torch.dtype)

    # Load Chronos
    pipeline = BaseChronosPipeline.from_pretrained(
        chronos_model_id,
        device_map=device,
        torch_dtype=torch_dtype,
    )

    if isinstance(pipeline, ChronosPipeline):
        predict_kwargs = dict(
            num_samples=num_samples,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
    elif isinstance(pipeline, ChronosBoltPipeline):
        predict_kwargs = {}

    # Load backtest configs
    with open(config_path) as fp:
        backtest_configs = yaml.safe_load(fp)

    result_rows = []
    for config in backtest_configs:
        dataset_name = config["name"]
        prediction_length = config["prediction_length"]

        logger.info(f"Loading {dataset_name}")
        test_data = load_and_split_dataset(backtest_config=config)

        logger.info(
            f"Generating forecasts for {dataset_name} "
            f"({len(test_data.input)} time series)"
        )
        forecasts = generate_forecasts(
            test_data.input,
            pipeline=pipeline,
            prediction_length=prediction_length,
            batch_size=batch_size,
            **predict_kwargs,
        )

        logger.info(f"Evaluating forecasts for {dataset_name}")
        metrics = (
            evaluate_forecasts(
                forecasts,
                test_data=test_data,
                metrics=[
                    MASE(),
                    MeanWeightedSumQuantileLoss(np.arange(0.1, 1.0, 0.1)),
                ],
                batch_size=5000,
            )
            .reset_index(drop=True)
            .to_dict(orient="records")
        )
        result_rows.append(
            {"dataset": dataset_name, "model": chronos_model_id, **metrics[0]}
        )

    # Save results to a CSV file
    results_df = (
        pd.DataFrame(result_rows)
        .rename(
            {"MASE[0.5]": "MASE", "mean_weighted_sum_quantile_loss": "WQL"},
            axis="columns",
        )
        .sort_values(by="dataset")
    )
    results_df.to_csv(metrics_path, index=False)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("Chronos Evaluation")
    logger.setLevel(logging.INFO)
    app()
