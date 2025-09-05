from pathlib import Path
import datasets
import numpy as np
import os
import pandas as pd

from constants import START_DATE, END_DATE, DATASETS_INFO
from covariates import steps, bells, arp
from target import single, simple, diverse, noisy


def make_positive(x):
    return x - x.min()


num_entries = 100


def generate():
    output_dir = Path("./datasets")
    output_dir.mkdir(exist_ok=True, parents=True)

    timestamp = np.array(
        pd.date_range(
            start=pd.to_datetime(START_DATE), end=pd.to_datetime(END_DATE), freq="D"
        )
    )

    for dataset_name, dataset_info in DATASETS_INFO.items():

        print(f"dataset_name:{dataset_name}")
        function_target = dataset_info.pop("function_target")
        function_covariates = dataset_info.pop("function_covariates")
        operator = dataset_info.pop("op")

        target_generator = eval(function_target)
        covariate_generator = eval(function_covariates)

        entries = []
        for _ in range(num_entries):
            target_before_covariates = target_generator(L=len(timestamp))
            target_after_covariates, covariate = covariate_generator(
                series=target_before_covariates,
                op=operator,
                function_target=function_target,
                **dataset_info,
            )
            target_after_covariates = make_positive(target_after_covariates)
            entry = {
                "timestamp": timestamp,
                "target": target_after_covariates,
                "covariate": covariate,
            }
            entries.append(entry)

        dest_file = os.path.join(output_dir, dataset_name)
        ds = datasets.Dataset.from_list(entries)
        ds.save_to_disk(dest_file)


if __name__ == "__main__":
    generate()
