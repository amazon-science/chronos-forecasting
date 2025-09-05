from pathlib import Path
from urllib import request
import datasets
import pandas as pd
import tempfile


MAIN_URL = "https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/f055eec3266eda77e183b62e9f07eff0bc6c155b/datasets/electricity.csv"
EX_URL = "https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/f055eec3266eda77e183b62e9f07eff0bc6c155b/datasets/exogenous-vars-electricity.csv"


def load_data(id: str):
    with tempfile.TemporaryDirectory() as dir_path:
        data_path = Path(dir_path) / "data.csv"

        request.urlretrieve(MAIN_URL, data_path)
        main_data = pd.read_csv(data_path)
        main_data = main_data[main_data.unique_id == id][["ds", "y"]]

        request.urlretrieve(EX_URL, data_path)
        ex_data = pd.read_csv(data_path)
        ex_data = ex_data[ex_data.unique_id == id]

    time_data = pd.date_range(
        start=main_data.ds.values[0], end=main_data.ds.values[-1], freq="H"
    )
    data = pd.merge(main_data, ex_data, how="left", on="ds")

    assert time_data.shape[0] == data.shape[0]

    return data


def generate(id: str):
    data = load_data(id)
    df = data[["ds", "y", "unique_id", "Exogenous1", "Exogenous2"]]
    df = df.rename(
        columns={
            "ds": "timestamp",
            "y": "target",
            "unique_id": "id",
            "Exogenous1": "exogenous1",
            "Exogenous2": "exogenous2",
        }
    )

    ds = datasets.Dataset.from_list([df.to_dict("list")])

    return ds


def get_epf_dataset(dataset_name: str):
    id = dataset_name.split("_")[-2].upper()
    ds = generate(id)

    return ds
