import webdataset as wds
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from tqdm import tqdm
import os

def get_batches(tarfile_path, schema):
    dataset = wds.WebDataset(tarfile_path, shardshuffle=False).decode("l")
    freq = f'{int(1000000/4096)}us'  # Should get freq as arg
    start_timestamp = pd.Timestamp("01-01-2019")  # Should get start timestamp from metadata
    for sample in tqdm(dataset):
        yield pa.RecordBatch.from_pydict(
                pd.DataFrame({
                "item_id": sample['index.npy'].item(), 
                "timestamp": pd.date_range(start_timestamp, periods=len(sample["segment.npy"]), freq=freq),
                "target": sample['segment.npy'].tolist(),
            }),
                schema=schema,
        )

def get_schema() -> pa.Schema:
    schema = pa.schema([
        ("item_id", pa.int32()),
        ("timestamp", pa.timestamp('us')),
        ("target", pa.float32())
    ])
    return schema

def get_patient_names():
    recording_list = pd.read_csv("/innovation_dataflow_dir/inputs/recording_list.epilepsiae_india.csv")
    all_patient_names = recording_list["patient_name"].unique().tolist()
    return all_patient_names

def main():
    schema = get_schema()
    patient_names = get_patient_names()
    input_dir = f"/innovation_cache/merged_eeg"
    output_dir = f"/innovation_cache/merged_eeg"
    for patient_name in patient_names:
        print(f"Processing {patient_name=}")
        filepath = f"{input_dir}/{patient_name}.univariate.tar"
        if not os.path.exists(filepath):
            print(f"{patient_name=} not exists, skipping...")
            continue
        try:
            with pq.ParquetWriter(f"{output_dir}/{patient_name}.univariate.parquet", schema=schema) as writer:
                for batch in get_batches(filepath, schema):
                    writer.write_batch(batch)
        except Exception as e:
            print(f"For {patient_name=}, got exception {e}")
            continue

if __name__ == "__main__":
    main()