import numpy as np
import yaml

from chronosx.chronosx import ChronosXPipeline
from chronosx.utils import ChronosDataset, load_and_split_dataset
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
from gluonts.model.evaluation import evaluate_forecasts
from pathlib import Path


config_path = "./configs/datasets.yaml"
output_dir = Path(f"../../output/finetune")
output_dir.mkdir(exist_ok=True, parents=True)

with open(config_path) as fp:
    backtest_configs = yaml.safe_load(fp)

dataset_config = backtest_configs[0]
prediction_length = dataset_config["prediction_length"]
num_covariates = 2 * len(dataset_config["covariates_fields"])


# Load Chronos
pipeline = ChronosXPipeline(
    prediction_length=prediction_length,
    num_covariates=num_covariates,
)

# load Dataset
train_dataset, test_dataset = load_and_split_dataset(backtest_config=dataset_config)
quantized_train_dataset = ChronosDataset(
    datasets=[train_dataset],
    probabilities=[1.0],
    tokenizer=pipeline.tokenizer,
    prediction_length=prediction_length,
    mode="training",
).shuffle()

# fine tune model
_, save_model_path = pipeline.finetune(
    output_dir,
    quantized_train_dataset,
    skip_pretrained_validation=True,
)

# Evaluate fine tuned model
pipeline = ChronosXPipeline(
    prediction_length=prediction_length,
    num_covariates=num_covariates,
    pretrained_model_name_or_path=output_dir / "final-checkpoint",
)

pipeline.chronosx.eval()
forecasts = pipeline.generate_forecasts(test_dataset.input)

metrics = (
    evaluate_forecasts(
        forecasts,
        test_data=test_dataset,
        metrics=[
            MASE(),
            MeanWeightedSumQuantileLoss(np.arange(0.05, 1, 0.05).round(2).tolist()),
        ],
    )
    .reset_index(drop=True)
    .to_dict(orient="records")
)

print(metrics)
