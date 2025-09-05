# Experiments in Real Datasets

An example on how to run experiments with real datasets is shown in [this script](./experiment_with_real_datasets.py).

The configurations per dataset are located in [this yaml file](./configs/datasets.yaml).

### On electricity price forecasting datasets
At the time of writing our paper we took the electricity price forecasting datasets from [here](https://github.com/Nixtla/transfer-learning-time-series/blob/main/datasets/exogenous-vars-electricity.csv) and [here](https://github.com/Nixtla/transfer-learning-time-series/blob/main/datasets/electricity.csv). As one can see from the commit history [here](https://github.com/Nixtla/transfer-learning-time-series/pull/17), it seems that there were some fixes on how the data was preprocessed. Since the data has been taken from [zenodo](https://zenodo.org/records/4624805), we have provided both versions of these datasets so that anyone interested can run our model with both the previous and corrected dataset versions.