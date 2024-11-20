import datasets
import pyarrow as pa
import pyarrow.feather as feather
import yaml
from os.path import join
import numpy as np
import pandas as pd
import json


def check_nan(df: pd.DataFrame):
    # Flatten the list column
    flattened_data = [item for sublist in df['target'] for item in sublist]

    # Count total numbers and NaN values
    total_numbers = len(flattened_data)
    nan_count = np.isnan(flattened_data).sum()

    # Calculate percentage of NaN values
    nan_percentage = (nan_count / total_numbers) * 100
    return nan_count, nan_percentage, total_numbers


def read_check_dataset(file_path, offset, file_path_val: str = None):
    # Reading the Arrow file
    with pa.memory_map(file_path, 'r') as source:
        table = feather.read_table(source)

    # Convert to a Pandas DataFrame if needed
    df = table.to_pandas()
    print(f'number of items:{len(df)}')
    print(f"len first array:{len(df['target'].iloc[0])}")
    number_samples_rows = []
    for sample in df['target']:
        number_samples_rows.append(len(sample))
    print(f'min samples:{np.min(number_samples_rows)}')
    print(f'mean samples:{np.mean(number_samples_rows)}')
    if file_path_val is None:
        print(
            f'number of samples with size less than 2*offset:{np.sum(np.array(number_samples_rows) < 2 * abs(offset) + 1)}')
    else:
        print(
            f'number of samples with size less than offset:{np.sum(np.array(number_samples_rows) < abs(offset) + 1)}')

    print(f'total number of samples rows:{len(number_samples_rows)}')

    if file_path_val:
        with pa.memory_map(file_path_val, 'r') as source:
            table = feather.read_table(source)

        # Convert to a Pandas DataFrame if needed
        df = table.to_pandas()
        print(f'val: number of items:{len(df)}')
        print(f"val: len first array:{len(df['target'].iloc[0])}")
        number_samples_rows = []
        for sample in df['target']:
            number_samples_rows.append(len(sample))
        print(f'val: min samples:{np.min(number_samples_rows)}')
        print(f'val: mean samples:{np.mean(number_samples_rows)}')
    return len(df)
    # nan_count, nan_percentage, total_numbers = check_nan(df)
    # print(f'{file_path}:nan_counts:{nan_count}; nans%: {nan_percentage}; total:{total_numbers}')


def extract_target_from_extra_datasets(ds):
    target = []
    for c in ds.column_names:
        if c not in ('id', 'timestamp'):
            target.append(ds[c])
    return np.vstack(target)


def preprocess_train_dataset(backtest_config: dict, save_path: str, create_val: bool = False,
                             save_path_val: str = None):
    print(f"dataset name:{backtest_config['name']}")
    print(f"the offset for current dataset:{backtest_config['offset']}")
    ds = datasets.load_dataset(backtest_config['hf_repo'], backtest_config['name'], split="train")
    print('dataset was loaded')
    ds.set_format("numpy")  # sequences returned as numpy arrays
    if 'target' in ds.column_names:
        target = ds['target']
        ids = ds['id']
        timestamps = ds['timestamp']
    else:
        # hardcoded because of two datasets in the 'extra' rep
        ids = np.tile(ds['id'], 7)
        timestamps = np.repeat(ds['timestamp'], 7, axis=0)
        target = extract_target_from_extra_datasets(ds)

    if target.ndim == 1:
        train_target_column = []
        if create_val:
            val_target_column = []
        start_column = []
        print(f"fist array len:{len(target[0])}")
        for arr in target:
            if create_val:
                train_target_column.append(arr[:2 * backtest_config['offset']].tolist())
                val_target_column.append(arr[3 * backtest_config['offset']:backtest_config['offset']].tolist())
            else:
                train_target_column.append(arr[:backtest_config['offset']].tolist())
        for arr in timestamps:
            start_column.append(arr[0])  # keeps only first date; otherwise it throws an error during the training
        # train_target_column = np.array(train_target_column)
    else:
        assert target.ndim == 2, "Supported only 1D or 2D arrays"
        print(f"the full dataset shape:{target.shape}")
        if create_val:
            train_target_column = target[:, :2 * backtest_config['offset']].tolist()
            val_target_column = target[:, 3 * backtest_config['offset']:backtest_config['offset']].tolist()
        else:
            train_target_column = target[:, :backtest_config['offset']].tolist()

        start_column = timestamps[:, 0]  # keeps only first date; otherwise it throws an error during the training
    # print(train_target_column)
    id_column = ids
    table = pa.table([id_column, start_column, pa.array(train_target_column)], names=['id', 'start', 'target'])
    if create_val:
        table_val = pa.table([id_column, start_column, pa.array(val_target_column)], names=['id', 'start', 'target'])
        with pa.OSFile(save_path_val, "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table_val.schema) as writer:
                writer.write_table(table_val)
    with pa.OSFile(save_path, "wb") as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)


if __name__ == '__main__':
    create_val = True
    save_dir = "zero_shot_datasets_train_val/"
    config_path = "../evaluation/configs/zero-shot.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    prediction_lengths = {}
    number_of_rows = {}
    for item in config:
        prediction_length = item['prediction_length']
        prediction_lengths[item['name']] = prediction_length
        print(f"dataset name:{item['name']}")
        print(f"offset number:{item['offset']}")
        if create_val:
            save_path = join(join(save_dir, 'train/'), item['name'] + '.arrow')
            save_path_val = join(join(save_dir, 'val/'), item['name'] + '.arrow')
            # preprocess_train_dataset(item, save_path, True, save_path_val)
            number_of_rows[item['name']] = read_check_dataset(save_path, item['offset'], save_path_val)
        else:
            save_path = join(save_dir, item['name'] + '.arrow')
            preprocess_train_dataset(item, save_path)
            read_check_dataset(save_path, item['offset'])

    json_save_path = 'prediction_lengths.json'
    json_save_path_number_of_rows = 'number_of_rows.json'

    # Write the prediction lengths to a JSON file
    with open(json_save_path, 'w') as json_file:
        json.dump(prediction_lengths, json_file, indent=4)

    # Write the prediction lengths to a JSON file
    with open(json_save_path_number_of_rows, 'w') as json_file:
        json.dump(number_of_rows, json_file, indent=4)
