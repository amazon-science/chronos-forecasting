from datasetsforecast.m5 import M5
from joblib import Parallel, delayed
from pathlib import Path
from tqdm.auto import tqdm
import datasets
import pandas as pd
import pyarrow as pa

def _process_entry(id_, ts, static_df):
    entry = {"id": id_}
    entry.update(ts.drop(columns=["id"]).to_dict("list"))
    entry.update(static_df.loc[id_].to_dict())
    return entry


def generate():
    target, covariates, static = M5.load(".")

    encoded_columns = pd.get_dummies(
        covariates[["event_type_1", "event_type_2"]]
    ).astype(float)

    covariates = covariates[["snap_CA", "snap_TX", "snap_WI", "sell_price"]]
    covariates = pd.concat([covariates, encoded_columns], axis=1)

    df = pd.concat([target, covariates], axis=1)
    df = df.rename(columns={"unique_id": "id", "ds": "timestamp", "y": "target"})
    df["timestamp"] = df["timestamp"].astype("datetime64[ms]")

    static = static.rename(columns={"unique_id": "id"}).set_index("id")

    processed = Parallel(n_jobs=-1)(
        delayed(_process_entry)(id_, ts, static) for id_, ts in tqdm(df.groupby("id"))
    )

    table = pa.Table.from_pylist(processed)

    splits = datasets.SplitDict()
    splits.add(datasets.SplitInfo('train', num_examples=len(processed)))
    info = datasets.DatasetInfo(splits=splits)

    dataset = datasets.Dataset(table, info=info, split='train')
    dataset.save_to_disk(Path(__file__).parent / "m5_with_covariates")


def get_m5_dataset(force_computation: bool = False):
    if (Path(__file__).parent / "m5_with_covariates").is_dir():
        if force_computation is True:
            generate()
    else:
        generate()

    return datasets.load_from_disk(Path(__file__).parent / "m5_with_covariates")


