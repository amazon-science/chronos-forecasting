import gc
import numpy as np
import pandas as pd
import random
import shutil
import time
import torch
import yaml


from chronosx.chronosx import ChronosXPipeline
from chronosx.utils.chronos_dataset import ChronosDataset
from chronosx.utils.hf_data_loader import load_and_split_dataset
from chronosx.utils.utils import has_enough_observations
from functools import partial
from gluonts.dataset.split import split
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
from gluonts.itertools import Filter
from gluonts.model.evaluation import evaluate_forecasts
from pathlib import Path
from pprint import pprint


covariate_injection = "IIB+OIB"
chronos_model_id = "amazon/chronos-t5-small"
config_path = "../configs/synthetic_datasets.yaml"
output_dir = Path("../../../output_synthetic_experiments/finetune")
output_metrics_dir = Path("../../../output_synthetic_experiments/metrics")
output_metrics_dir.mkdir(exist_ok=True, parents=True)

min_past = 1
shuffle_buffer_length = 100
num_runs = 3
max_steps = 5000
skip_pretrained_validation = False
learning_rate_list = [0.01, 0.001, 0.0001]

with open(config_path) as fp:
    backtest_configs = yaml.safe_load(fp)

for dataset_config in backtest_configs:
    dataset_name = dataset_config["name"]
    prediction_length = dataset_config["prediction_length"]
    num_covariates = 2 * len(dataset_config["covariates_fields"])
    train_dataset, test_dataset = load_and_split_dataset(backtest_config=dataset_config)

    # Load Chronos
    pipeline = ChronosXPipeline(
        prediction_length=prediction_length,
        num_covariates=num_covariates,
        covariate_injection=covariate_injection,
        pretrained_model_name_or_path=chronos_model_id,
    )

    tokenizer = pipeline.tokenizer

    train_dataset = Filter(
        partial(
            has_enough_observations,
            min_length=min_past + 2 * prediction_length,  # for validation and for train
            max_missing_prop=0.5,
        ),
        train_dataset,
    )

    quantized_val_dataset = ChronosDataset(
        datasets=[train_dataset],
        probabilities=[1.0],
        tokenizer=tokenizer,
        prediction_length=prediction_length,
        min_past=min_past,
        mode="validation",
    )

    train_dataset, _ = split(train_dataset, offset=-prediction_length)
    quantized_train_dataset = ChronosDataset(
        datasets=[train_dataset],
        probabilities=[1.0],
        tokenizer=tokenizer,
        prediction_length=prediction_length,
        min_past=min_past,
        mode="training",
    ).shuffle(shuffle_buffer_length=shuffle_buffer_length)

    # zero-shot evaluation on validation set
    val_of_zero_shot_pretrained_model = pipeline.evaluate_model_on_validation_set(
        covariate_injection=None,
        quantized_val_dataset=quantized_val_dataset,
        output_dir=output_dir / dataset_name,
    )

    random.seed(int(time.time()))
    seed = random.randint(0, 2**32)
    val_loss_per_lr = {}
    mean_val_loss_per_lr = {}

    for lr in learning_rate_list:
        lr_str = str(lr)
        val_loss_per_run = []
        model_paths = []
        for run_id in range(num_runs):
            run_id_str = str(run_id)
            output_dir_lr_run_id = Path(
                output_dir / dataset_name / f"lr={lr}" / f"run_id={run_id}"
            )
            output_dir_lr_run_id.mkdir(exist_ok=True, parents=True)

            quantized_train_dataset = ChronosDataset(
                datasets=[train_dataset],
                probabilities=[1.0],
                tokenizer=tokenizer,
                prediction_length=prediction_length,
                min_past=min_past,
                mode="training",
            ).shuffle(
                shuffle_buffer_length=shuffle_buffer_length,
                random_seed=run_id + seed,
            )

            pipeline = ChronosXPipeline(
                prediction_length=prediction_length,
                num_covariates=num_covariates,
                covariate_injection=covariate_injection,
                pretrained_model_name_or_path=chronos_model_id,
            )

            val_loss, save_model_path = pipeline.finetune(
                output_dir_lr_run_id,
                quantized_train_dataset,
                lr=lr,
                quantized_val_dataset=quantized_val_dataset,
                skip_pretrained_validation=skip_pretrained_validation,
                max_steps=max_steps,
            )

            val_loss_per_run.append(val_loss)
            model_paths.append(save_model_path)

        mean_val_loss = np.mean(val_loss_per_run)
        val_loss_per_lr[f"{lr}"] = val_loss_per_run
        mean_val_loss_per_lr[f"{lr}"] = mean_val_loss
        print(
            f"lr: {lr} - val_loss_per_run: {val_loss_per_run} - mean_val_loss: {mean_val_loss}"
        )

    print("val_loss_per_lr")
    pprint(val_loss_per_lr)

    print("mean_val_loss_per_lr")
    pprint(mean_val_loss_per_lr)

    best_val_loss = np.inf
    for lr_str, val_loss in mean_val_loss_per_lr.items():
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_lr_str = lr_str

    # evaluate model with best mean validation loss
    lr = float(best_lr_str)
    result_rows = []
    for run_id in range(num_runs):
        pretrained_model_path = (
            output_dir
            / dataset_name
            / f"lr={lr}"
            / f"run_id={run_id}"
            / "final-checkpoint"
        )
        pipeline = ChronosXPipeline(
            prediction_length=prediction_length,
            num_covariates=num_covariates,
            covariate_injection=covariate_injection,
            pretrained_model_name_or_path=pretrained_model_path,
        )

        pipeline.chronosx.eval()
        forecasts = pipeline.generate_forecasts(
            test_dataset.input,
        )

        metrics = (
            evaluate_forecasts(
                forecasts,
                test_data=test_dataset,
                metrics=[
                    MASE(),
                    MeanWeightedSumQuantileLoss(
                        np.arange(0.05, 1, 0.05).round(2).tolist()
                    ),
                ],
                batch_size=5000,
            )
            .reset_index(drop=True)
            .to_dict(orient="records")
        )

        result_rows.append(
            {
                "dataset": dataset_config["name"],
                "covariate_injection": covariate_injection,
                **metrics[0],
            }
        )

    df = pd.DataFrame(result_rows)
    df = df.set_index(["dataset", "covariate_injection"])
    pprint(df.mean())
    df.to_csv(output_metrics_dir / f"{dataset_name.split('/')[-1]}_max_steps={max_steps}.csv")

    shutil.rmtree(output_dir / dataset_name)
    del pipeline, quantized_train_dataset, quantized_val_dataset, forecasts
    gc.collect()
    torch.cuda.empty_cache()
