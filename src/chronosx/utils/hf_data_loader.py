import datasets
import numpy as np
import pandas as pd
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.split import split
from typing import List
from .m5 import get_m5_dataset
from .epf import get_epf_dataset


def to_gluonts_univariate(
    hf_dataset: datasets.Dataset,
    series_fields: List[str],
    covariates_fields: List[str] = None,
):
    if isinstance(series_fields, str):
        series_fields = [series_fields]

    # dataset_length = hf_dataset.info.splits["train"].num_examples * len(series_fields)
    dataset_length = len(hf_dataset) * len(series_fields)

    # Assumes that all time series in the dataset have the same frequency
    dataset_freq = pd.DatetimeIndex(hf_dataset[0]["timestamp"]).to_period()[0].freqstr

    gts_dataset = []
    for hf_entry in hf_dataset:
        for field in series_fields:
            entry = {
                "start": pd.Period(
                    hf_entry["timestamp"][0],
                    freq=dataset_freq,
                ),
                "target": hf_entry[field],
            }

            if covariates_fields:
                covariates = np.array([hf_entry[field] for field in covariates_fields])
                is_nan_covariates = np.isnan(covariates)
                covariates[is_nan_covariates] = -1
                covariates = np.vstack([covariates, is_nan_covariates]).astype(
                    np.float32
                )
                entry.update({FieldName.FEAT_DYNAMIC_REAL: covariates})

            gts_dataset.append(entry)

    assert len(gts_dataset) == dataset_length

    return gts_dataset


def load_and_split_dataset(backtest_config: dict):
    hf_repo = backtest_config.get("hf_repo", None)
    filename = backtest_config.get("filename", None)
    dataset_name = backtest_config["name"]
    offset = backtest_config["offset"]
    prediction_length = backtest_config["prediction_length"]
    num_rolls = backtest_config["num_rolls"]
    series_fields = backtest_config.get("series_fields")
    covariates_fields = backtest_config.get("covariates_fields")

    if dataset_name == "m5_with_covariates":
        ds = get_m5_dataset()

    elif dataset_name in [
        "epf_electricity_be_paper",
        "epf_electricity_de_paper",
        "epf_electricity_fr_paper",
        "epf_electricity_np_paper",
        "epf_electricity_pjm_paper",
    ]:
        ds = get_epf_dataset(dataset_name)

    elif 'synthetic_datasets' in dataset_name:
        ds = datasets.load_from_disk(filename)

    else:
        ds = datasets.load_dataset(
            hf_repo, dataset_name, split="train", trust_remote_code=True
        )

    ds.set_format("numpy")

    gts_dataset = to_gluonts_univariate(
        ds, covariates_fields=covariates_fields, series_fields=series_fields
    )

    # Split dataset for evaluation
    train_dataset, test_template = split(gts_dataset, offset=offset)
    test_data = test_template.generate_instances(prediction_length, windows=num_rolls)

    return train_dataset, test_data
