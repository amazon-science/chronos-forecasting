# Usage Examples

## Generating Synthetic Time Series (KernelSynth)

- Install this package with with the `training` extra:
    ```
    pip install "chronos[training] @ git+https://github.com/amazon-science/chronos-forecasting.git"
    ```
- Run `kernel-synth.py`:
    ```sh
    # With defaults used in the paper (1M time series and 5 max_kernels)
    python kernel-synth.py

    # You may optionally specify num-series and max-kernels
    python kernel-synth.py \
        --num-series <num of series to generate> \
        --max-kernels <max number of kernels to use per series>
    ```
    The generated time series will be saved in a [GluonTS](https://github.com/awslabs/gluonts)-comptabile arrow file `kernelsynth-data.arrow`.

## Pretraining (and fine-tuning) Chronos models
- Install this package with with the `training` extra:
    ```
    pip install "chronos[training] @ git+https://github.com/amazon-science/chronos-forecasting.git"
    ```
- Convert your time series dataset into a GluonTS-compatible file dataset. We recommend using the arrow format. You may use the `convert_to_arrow` function from the following snippet for that. Optionally, you may use [synthetic data from KernelSynth](#generating-synthetic-time-series-kernelsynth) to follow along.
    ```py
    from pathlib import Path
    from typing import List, Optional, Union

    import numpy as np
    from gluonts.dataset.arrow import ArrowWriter


    def convert_to_arrow(
        path: Union[str, Path],
        time_series: Union[List[np.ndarray], np.ndarray],
        start_times: Optional[Union[List[np.datetime64], np.ndarray]] = None,
        compression: str = "lz4",
    ):
        if start_times is None:
            # Set an arbitrary start time
            start_times = [np.datetime64("2000-01-01 00:00", "s")] * len(time_series)

        assert len(time_series) == len(start_times)

        dataset = [
            {"start": start, "target": ts} for ts, start in zip(time_series, start_times)
        ]
        ArrowWriter(compression=compression).write_to_file(
            dataset,
            path=path,
        )


    if __name__ == "__main__":
        # Generate 20 random time series of length 1024
        time_series = [np.random.randn(1024) for i in range(20)]

        # Convert to GluonTS arrow format
        convert_to_arrow("./noise-data.arrow", time_series=time_series)

    ```
- Modify the [training configs](training/configs) to use your data. Let's use the KernelSynth data as an example.
    ```yaml
    # List of training data files
    training_data_paths:
    - "/path/to/kernelsynth-data.arrow"
    # Mixing probability of each dataset file
    probability:
    - 1.0
    ```
    You may optionally change other parameters of the config file, as required. For instance, if you're interested in fine-tuning the model from a pretrained Chronos checkpoint, you should change the `model_id`, set `random_init: false`, and (optionally) change other parameters such as `max_steps` and `learning_rate`.
- Start the training (or fine-tuning) job:
    ```sh
    # On single GPU
    CUDA_VISIBLE_DEVICES=0 python training/train.py --config/path/to/modified/config.yaml

    # On multiple GPUs (example with 8 GPUs)
    torchrun --nproc-per-node=8 training/train.py --config /path/to/modified/config.yaml

    # Fine-tune `amazon/chronos-t5-small` for 1000 steps
    CUDA_VISIBLE_DEVICES=0 python training/train.py --config /path/to/modified/config.yaml \
        --model-id amazon/chronos-t5-small \
        --no-random-init \
        --max-steps 1000
    ```
    The output and checkpoints will be saved in `output/run_{id}/`.
> [!TIP]  
> If the initial training step is too slow, you might want to change the `shuffle_buffer_length` and/or set `torch_compile` to `false`.