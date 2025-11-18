# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Authors: Abdul Fatir Ansari <ansarnd@amazon.com>

import logging
import math
import time
import warnings
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
import torch
from einops import rearrange, repeat
from torch.utils.data import DataLoader
from transformers import AutoConfig

import chronos.chronos2
from chronos.base import BaseChronosPipeline, ForecastType
from chronos.chronos2 import Chronos2Model
from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode, TensorOrArray
from chronos.df_utils import convert_df_input_to_list_of_dicts_input
from chronos.utils import interpolate_quantiles, weighted_quantile

if TYPE_CHECKING:
    import datasets
    import fev
    import pandas as pd

logger = logging.getLogger(__name__)


class Chronos2Pipeline(BaseChronosPipeline):
    forecast_type: ForecastType = ForecastType.QUANTILES
    default_context_length: int = 2048

    def __init__(self, model: Chronos2Model):
        super().__init__(inner_model=model)
        self.model = model

    @staticmethod
    def _get_prob_mass_per_quantile_level(quantile_levels: torch.Tensor) -> torch.Tensor:
        """
        Computes normalized probability masses for quantile levels using trapezoidal rule approximation.

        Each quantile receives probability mass proportional to the width of its surrounding interval,
        creating a piecewise uniform distribution. The mass for quantile q_i is computed as
        (q_{i+1} - q_{i-1}) / 2, where q_0 = 0 and q_{n+1} = 1.

        Parameters
        ----------
        quantile_levels : torch.Tensor
            The quantile levels, must be strictly in (0, 1)

        Returns
        -------
        torch.Tensor
            The normalized probability mass per quantile
        """
        assert quantile_levels.ndim == 1
        assert quantile_levels.min() > 0.0 and quantile_levels.max() < 1.0

        device = quantile_levels.device
        boundaries = torch.cat(
            [torch.tensor([0.0], device=device), quantile_levels, torch.tensor([1.0], device=device)]
        )
        prob_mass = (boundaries[2:] - boundaries[:-2]) / 2
        return prob_mass / prob_mass.sum()

    @property
    def model_context_length(self) -> int:
        return self.model.chronos_config.context_length

    @property
    def model_output_patch_size(self) -> int:
        return self.model.chronos_config.output_patch_size

    @property
    def model_prediction_length(self) -> int:
        return self.model.chronos_config.max_output_patches * self.model.chronos_config.output_patch_size

    @property
    def quantiles(self) -> list[float]:
        return self.model.chronos_config.quantiles

    @property
    def max_output_patches(self) -> int:
        return self.model.chronos_config.max_output_patches

    def fit(
        self,
        inputs: TensorOrArray
        | Sequence[TensorOrArray]
        | Sequence[Mapping[str, TensorOrArray | Mapping[str, TensorOrArray | None]]],
        prediction_length: int,
        validation_inputs: TensorOrArray
        | Sequence[TensorOrArray]
        | Sequence[Mapping[str, TensorOrArray | Mapping[str, TensorOrArray | None]]]
        | None = None,
        context_length: int | None = None,
        learning_rate: float = 1e-6,
        num_steps: int = 1000,
        batch_size: int = 256,
        output_dir: Path | str | None = None,
        min_past: int | None = None,
        finetuned_ckpt_name: str = "finetuned-ckpt",
        **extra_trainer_kwargs,
    ) -> "Chronos2Pipeline":
        """
        Fine-tune a copy of the current Chronos-2 model on the given inputs and return a new pipeline.

        Parameters
        ----------
        inputs
            The time series on which the model will be fine-tuned. The allowed formats of inputs are the same as `Chronos2Pipeline.predict()`.
            Note: when `inputs` is a list of dicts, the values inside `future_covariates` are not technically used for training the model;
            however, this key is used to infer which covariates are known into the future. Therefore, if your task contains known future covariates,
            make sure that this key exists in `inputs`. The values of individual future covariates may be set to `None` or an empty array.
        prediction_length
            The prediction horizon for which the model will be fine-tuned
        validation_inputs
            The time series used for validation and model selection. The format of `validation_inputs` is exactly the same as `inputs`, by default None which
            means that no validation is performed. Note that enabling validation may slow down fine-tuning for large datasets.
        context_length
            The maximum context length used during fine-tuning, by default set to the model's default context length
        learning_rate
            The learning rate for the optimizer, by default 1e-6
        num_steps
            The number of steps to fine-tune for, by default 1000
        batch_size
            The batch size used for fine-tuning. Note that the batch size here means the number of time series, including target(s) and covariates,
            which are input into the model. If your data has multiple target and/or covariates, the effective number of time series tasks in a batch
            will be lower than this value, by default 256
        output_dir
            The directory in which outputs from the `Trainer` will be saved, by default set to `chronos-2-finetuned/{%Y-%m-%d_%H-%M-%S}`
        min_past
            The minimum number of time steps the context must have during fine-tuning. All time series shorter than `min_past + prediction_length`
            are filtered out, by default set equal to prediction_length
        finetuned_ckpt_name
            The name of the directory inside `output_dir` in which the final fine-tuned checkpoint will be saved, by default "finetuned-ckpt"
        **extra_trainer_kwargs
            Extra kwargs are directly forwarded to `TrainingArguments`

        Returns
        -------
        A new `Chronos2Pipeline` with the fine-tuned model
        """

        import torch.cuda
        from transformers.training_args import TrainingArguments

        from chronos.chronos2.trainer import Chronos2Trainer, EvaluateAndSaveFinalStepCallback

        # Create a copy of the model to avoid modifying the original
        config = deepcopy(self.model.config)
        model = Chronos2Model(config).to(self.model.device)  # type: ignore
        model.load_state_dict(self.model.state_dict())

        if context_length is None:
            context_length = self.model_context_length

        if min_past is None:
            min_past = prediction_length

        train_dataset = Chronos2Dataset.convert_inputs(
            inputs=inputs,
            context_length=context_length,
            prediction_length=prediction_length,
            batch_size=batch_size,
            output_patch_size=self.model_output_patch_size,
            min_past=min_past,
            mode=DatasetMode.TRAIN,
        )

        if output_dir is None:
            output_dir = Path("chronos-2-finetuned") / time.strftime("%Y-%m-%d_%H-%M-%S")
        elif isinstance(output_dir, str):
            output_dir = Path(output_dir)

        assert isinstance(output_dir, Path)

        use_cpu = str(self.model.device) == "cpu"
        has_sm80 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

        # warn user if a cuda device is available and CPU fine-tuning is used
        if use_cpu and torch.cuda.is_available():
            warnings.warn(
                "The model is being fine-tuned on the CPU, but a CUDA device is available. "
                "We recommend using the GPU for faster fine-tuning.",
                category=UserWarning,
                stacklevel=2,
            )

        training_kwargs: dict = dict(
            output_dir=str(output_dir),
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            lr_scheduler_type="linear",
            warmup_ratio=0.0,
            optim="adamw_torch_fused",
            logging_strategy="steps",
            logging_steps=100,
            disable_tqdm=False,
            report_to="none",
            max_steps=num_steps,
            gradient_accumulation_steps=1,
            dataloader_num_workers=1,
            tf32=has_sm80 and not use_cpu,
            bf16=has_sm80 and not use_cpu,
            save_only_model=True,
            prediction_loss_only=True,
            save_total_limit=1,
            save_strategy="no",
            save_steps=None,
            eval_strategy="no",
            eval_steps=None,
            load_best_model_at_end=False,
            metric_for_best_model=None,
            use_cpu=use_cpu,
        )

        eval_dataset = None
        callbacks = []
        if validation_inputs is not None:
            # construct validation dataset
            eval_dataset = Chronos2Dataset.convert_inputs(
                inputs=validation_inputs,
                context_length=context_length,
                prediction_length=prediction_length,
                batch_size=batch_size,
                output_patch_size=self.model_output_patch_size,
                mode=DatasetMode.VALIDATION,
            )

            # set validation parameters
            training_kwargs["save_strategy"] = "steps"
            training_kwargs["save_steps"] = 100
            training_kwargs["eval_strategy"] = "steps"
            training_kwargs["eval_steps"] = 100
            training_kwargs["load_best_model_at_end"] = True
            training_kwargs["metric_for_best_model"] = "eval_loss"
            training_kwargs["label_names"] = ["future_target"]

            # add callback to ensure that the final model is evaluated
            callbacks.append(EvaluateAndSaveFinalStepCallback())

        training_kwargs.update(extra_trainer_kwargs)

        if training_kwargs["tf32"]:
            # setting tf32=True changes these global properties, we copy them here so that
            # we can restore them after fine-tuning
            matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
            cudnn_tf32 = torch.backends.cudnn.allow_tf32

        training_args = TrainingArguments(**training_kwargs)

        trainer = Chronos2Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
        )
        trainer.train()

        # update max_output_patches, if the model was fine-tuned with longer prediction_length
        model.chronos_config.max_output_patches = max(
            model.chronos_config.max_output_patches, math.ceil(prediction_length / self.model_output_patch_size)
        )

        # Create a new pipeline with the fine-tuned model
        finetuned_pipeline = Chronos2Pipeline(model=model)

        # Save fine-tuned model
        finetuned_path = output_dir / finetuned_ckpt_name
        finetuned_pipeline.save_pretrained(finetuned_path)
        logger.info(f"Finetuned model saved to {finetuned_path}")

        if training_kwargs["tf32"]:
            # restore tf32 settings
            torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
            torch.backends.cudnn.allow_tf32 = cudnn_tf32

        return finetuned_pipeline

    def _prepare_inputs_for_long_horizon_unrolling(
        self,
        context: torch.Tensor,
        group_ids: torch.Tensor,
        future_covariates: torch.Tensor,
        unrolled_quantiles: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Expand the context, future_covariates and group_ids along a new "quantile" axis
        if future_covariates is not None:
            future_covariates = repeat(future_covariates, "b t -> b q t", q=len(unrolled_quantiles))
        context = repeat(context, "b t -> b q t", q=len(unrolled_quantiles))
        group_ids = repeat(group_ids, "b -> b q", q=len(unrolled_quantiles))
        # Shift the group_ids so that mixing is enabled only for time series with the same group_id and
        # at the same quantile level, e.g., if the group_ids were [0, 0, 1, 1, 1] initially, after expansion
        # and shifting they will be:
        # [[0,  1,  2,  3,  4,  5,  6,  7,  8],
        #  [0,  1,  2,  3,  4,  5,  6,  7,  8],
        #  [9, 10, 11, 12, 13, 14, 15, 16, 17],
        #  [9, 10, 11, 12, 13, 14, 15, 16, 17],
        #  [9, 10, 11, 12, 13, 14, 15, 16, 17]]
        group_ids = group_ids * len(unrolled_quantiles) + torch.arange(
            len(unrolled_quantiles), device=self.model.device
        ).unsqueeze(0)
        # We unroll the quantiles in unrolled_quantiles to the future and each unrolled quantile gives
        # len(self.quantiles) predictions, so we end up with len(unrolled_quantiles) * len(self.quantiles)
        # "samples". unrolled_sample_weights specifies the amount of probability mass covered by each sample.
        # Note that this effectively leads to shrinking of the probability space but it is better heuristic
        # than just using the median to unroll, which leads to uncertainty collapse.
        unrolled_sample_weights = torch.outer(
            self._get_prob_mass_per_quantile_level(unrolled_quantiles),
            self._get_prob_mass_per_quantile_level(torch.tensor(self.quantiles)),
        )

        return context, group_ids, future_covariates, unrolled_sample_weights

    def _autoregressive_unroll_for_long_horizon(
        self,
        context: torch.Tensor,
        group_ids: torch.Tensor,
        future_covariates: torch.Tensor,
        prediction: torch.Tensor,
        unrolled_quantiles: torch.Tensor,
        unrolled_sample_weights: torch.Tensor,
        num_output_patches: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get unrolled_quantiles from prediction and append it to the expanded context
        prediction_unrolled = interpolate_quantiles(
            query_quantile_levels=unrolled_quantiles,
            original_quantile_levels=self.quantiles,
            original_values=rearrange(prediction, "b q h -> b h q"),
        )
        prediction_unrolled = rearrange(prediction_unrolled, "b h q -> b q h")
        context = torch.cat([context, prediction_unrolled], dim=-1)[..., -self.model_context_length :]
        n_paths = len(unrolled_quantiles)

        # Shift future_covariates by prediction.shape[-1] while replacing the predicted values
        # of future covariates in the context with their actual values, if known
        if future_covariates is not None:
            context, future_covariates = self._slide_context_and_future_covariates(
                context=context, future_covariates=future_covariates, slide_by=prediction.shape[-1]
            )

        # Reshape (batch, n_paths, context_length) -> (batch * n_paths, context_length)
        prediction = self._predict_step(
            context=rearrange(context, "b n t -> (b n) t"),
            future_covariates=rearrange(future_covariates, "b n t -> (b n) t")
            if future_covariates is not None
            else None,
            group_ids=rearrange(group_ids, "b n -> (b n)"),
            num_output_patches=num_output_patches,
        )
        # Reshape predictions from (batch * n_paths, n_quantiles, length) to (batch, n_paths * n_quantiles, length)
        prediction = rearrange(prediction, "(b n) q h -> b (n q) h", n=n_paths)
        # Reduce `n_paths * n_quantiles` to n_quantiles and transpose back
        prediction = weighted_quantile(
            query_quantile_levels=self.quantiles,
            sample_weights=rearrange(unrolled_sample_weights, "n q -> (n q)"),
            samples=rearrange(prediction, "b (n q) h -> b h (n q)", n=n_paths),
        )
        prediction = rearrange(prediction, "b h q -> b q h")

        return prediction, context, future_covariates

    @torch.no_grad()
    def predict(
        self,
        inputs: TensorOrArray
        | Sequence[TensorOrArray]
        | Sequence[Mapping[str, TensorOrArray | Mapping[str, TensorOrArray]]],
        prediction_length: int | None = None,
        batch_size: int = 256,
        context_length: int | None = None,
        predict_batches_jointly: bool = False,
        limit_prediction_length: bool = False,
        **kwargs,
    ) -> list[torch.Tensor]:
        """
        Generate forecasts for the given time series.

        Parameters
        ----------
        inputs
            The time series to generate forecasts for, can be one of:
            - A 3-dimensional `torch.Tensor` or `np.ndarray` of shape (batch, n_variates, history_length). When `n_variates > 1`, information
            will be shared among the different variates of each time series in the batch and the model will perform multivariate forecasting.
            - A list of `torch.Tensor` or `np.ndarray` where each element can either be 1-dimensional of shape (history_length,)
            or 2-dimensional of shape (n_variates, history_length). The history_lengths may be different across elements; left-padding
            will be applied, if needed. The model will perform univariate and multivariate inference for 1-d and 2-d elements, respectively.
            A mixture of 1-d and 2-d elements can be provided in the same list.
            - A list of dictionaries where each dictionary may have the following keys.
                1. `target` (required): a 1-d or 2-d `torch.Tensor` or `np.ndarray` of shape (history_length,) or (n_variates, history_length).
                Forecasts will be generated for items in `target`.
                2. `past_covariates` (optional): a dict of past-only covariates or past values of known future covariates. The keys of the dict
                must be names of the covariates and values must be 1-d `torch.Tensor` or `np.ndarray` with length equal to the `history_length`
                of `target`.
                3. `future_covariates` (optional): a dict of future values of known future covariates. The keys of the dict must be names of the
                covariates and values must be 1-d `torch.Tensor` or `np.ndarray` with length equal to the `prediction_length`. All keys in
                `future_covariates` must be a subset of the keys in `past_covariates`.

            Examples:
            ```python

            # Batch of univariate time series
            inputs = torch.randn(32, 1, 100)

            # Batch of multivariate time series
            inputs = torch.randn(32, 3, 100)

            # List of time series with different lengths and n_variates
            inputs = [
                torch.randn(100),  # univariate series of length 100
                torch.randn(2, 150),  # bivariate series of length 150
                torch.randn(120),  # univariate series of length 120
            ]

            # List of dictionaries with covariates
            prediction_length = 24
            inputs = [
                {
                    # task with 1-d target, one past-only covariate and one known future covariate
                    "target": torch.randn(100),
                    "past_covariates": {"temperature": torch.randn(100), "precipitation": torch.randn(100)},
                    "future_covariates": {"temperature": torch.randn(prediction_length)},
                },
                {
                    # task with 2-d target and one past-only covariate
                    "target": torch.randn(2, 150),
                    "past_covariates": {"wind_speed": torch.randn(150)},
                },
                {
                    # task with 1-d target, two numeric covariates one of which is known into the future
                    # and one categorical covariate known into the future
                    # Note: categorical covariates are only supported as numpy arrays as torch does not support str dtype
                    "target": np.random.randn(150),
                    "past_covariates": {
                        "numeric_covariate_1": np.random.rand(150),
                        "numeric_covariate_2": np.random.rand(150),
                        "cat_covariate": np.random.choice(["A", "B", "C", "D", "E"], size=150),
                    },
                    "future_covariates": {
                        "numeric_covariate_1": np.random.rand(prediction_length),
                        "cat_covariate": np.random.choice(["A", "B", "C", "D", "E"], size=prediction_length),
                    },
                },
                {
                    # task with only a 1-d target
                    "target": torch.randn(1, 150)
                },
            ]
            ```
        prediction_length
            The number of time steps to predict for, defaults to the model's default prediction length
        batch_size
            The batch size used for prediction. Note that the batch size here means the number of time series, including target(s) and covariates,
            which are input into the model. If your data has multiple target and/or covariates, the effective number of time series tasks in a batch
            will be lower than this value, by default 256
        context_length
            The maximum context length used during for inference, by default set to the model's default context length
        predict_batches_jointly
            If True, cross-learning is enabled, i.e., all the tasks in `inputs` will be predicted jointly and the model will share information across all inputs, by default False
            The following must be noted when using cross-learning:
            - Cross-learning doesn't always improve forecast accuracy and must be tested for individual use cases.
            - Results become dependent on batch size. Very large batch sizes may not provide benefits as they deviate from the maximum group size used during pretraining.
            For optimal results, consider using a batch size around 100 (as used in the Chronos-2 technical report).
            - Cross-learning is most helpful when individual time series have limited historical context, as the model can leverage patterns from related series in the batch.
        limit_prediction_length
            If True, an error is raised when prediction_length is greater than model's default prediction length, by default False

        Returns
        -------
        The model's predictions, a list of `torch.Tensor` where each element has shape (n_variates, n_quantiles, prediction_length) and the number of
        elements are equal to the number of target time series (univariate or multivariate) in the `inputs`.

        """
        model_prediction_length = self.model_prediction_length
        if prediction_length is None:
            prediction_length = model_prediction_length

        # The maximum number of output patches to generate in a single forward pass before the long-horizon heuristic kicks in. Note: A value larger
        # than the model's default max_output_patches may lead to degradation in forecast accuracy, defaults to a model-specific value
        max_output_patches = kwargs.pop("max_output_patches", self.max_output_patches)
        # The set of quantiles to use when making long-horizon predictions; must be a subset of the model's default quantiles. These quantiles
        # are appended to the historical context and input into the model autoregressively to generate long-horizon predictions. Note that the
        # effective batch size increases by a factor of `len(unrolled_quantiles)` when making long-horizon predictions,
        # by default [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        unrolled_quantiles = kwargs.pop("unrolled_quantiles", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        if len(kwargs) > 0:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs.keys())}.")

        if not set(unrolled_quantiles).issubset(self.quantiles):
            raise ValueError(
                f"Unrolled quantiles must be a subset of the model's quantiles. "
                f"Found: {unrolled_quantiles=}, model_quantiles={self.quantiles}"
            )
        unrolled_quantiles_tensor = torch.tensor(unrolled_quantiles)

        if prediction_length > model_prediction_length:
            msg = (
                f"We recommend keeping prediction length <= {model_prediction_length}. "
                "The quality of longer predictions may degrade since the model is not optimized for it. "
            )
            if limit_prediction_length:
                msg += "You can turn off this check by setting `limit_prediction_length=False`."
                raise ValueError(msg)
            warnings.warn(msg)

        if context_length is None:
            context_length = self.model_context_length

        if context_length > self.model_context_length:
            warnings.warn(
                f"The specified context_length {context_length} is greater than the model's default context length {self.model_context_length}. "
                f"Resetting context_length to {self.model_context_length}."
            )
            context_length = self.model_context_length

        test_dataset = Chronos2Dataset.convert_inputs(
            inputs=inputs,
            context_length=context_length,
            prediction_length=prediction_length,
            batch_size=batch_size,
            output_patch_size=self.model_output_patch_size,
            mode=DatasetMode.TEST,
        )
        test_loader = DataLoader(test_dataset, batch_size=None, pin_memory=True, shuffle=False, drop_last=False)

        all_predictions: list[torch.Tensor] = []
        for batch in test_loader:
            assert batch["future_target"] is None
            batch_context = batch["context"]
            batch_group_ids = batch["group_ids"]
            batch_future_covariates = batch["future_covariates"]
            batch_target_idx_ranges = batch["target_idx_ranges"]

            if predict_batches_jointly:
                batch_group_ids = torch.zeros_like(batch_group_ids)

            batch_prediction = self._predict_batch(
                context=batch_context,
                group_ids=batch_group_ids,
                future_covariates=batch_future_covariates,
                unrolled_quantiles_tensor=unrolled_quantiles_tensor,
                prediction_length=prediction_length,
                max_output_patches=max_output_patches,
                target_idx_ranges=batch_target_idx_ranges,
            )
            all_predictions.extend(batch_prediction)

        return all_predictions

    def _predict_batch(
        self,
        context: torch.Tensor,
        group_ids: torch.Tensor,
        future_covariates: torch.Tensor,
        unrolled_quantiles_tensor: torch.Tensor,
        prediction_length: int,
        max_output_patches: int,
        target_idx_ranges: list[tuple[int, int]],
    ) -> list[torch.Tensor]:
        context = context.to(device=self.model.device, dtype=torch.float32)
        group_ids = group_ids.to(device=self.model.device)
        future_covariates = future_covariates.to(device=self.model.device, dtype=torch.float32)

        def get_num_output_patches(remaining_horizon: int):
            num_output_patches = math.ceil(remaining_horizon / self.model_output_patch_size)
            num_output_patches = min(num_output_patches, max_output_patches)

            return num_output_patches

        predictions = []
        remaining = prediction_length

        # predict first set of patches up to max_output_patches
        prediction: torch.Tensor = self._predict_step(
            context=context,
            group_ids=group_ids,
            future_covariates=future_covariates,
            num_output_patches=get_num_output_patches(remaining),
        )
        predictions.append(prediction)
        remaining -= prediction.shape[-1]

        # prepare inputs for long horizon prediction
        if remaining > 0:
            context, group_ids, future_covariates, unrolled_sample_weights = (
                self._prepare_inputs_for_long_horizon_unrolling(
                    context=context,
                    group_ids=group_ids,
                    future_covariates=future_covariates,
                    unrolled_quantiles=unrolled_quantiles_tensor,
                )
            )

        # long horizon heuristic
        while remaining > 0:
            prediction, context, future_covariates = self._autoregressive_unroll_for_long_horizon(
                context=context,
                group_ids=group_ids,
                future_covariates=future_covariates,
                prediction=prediction,
                unrolled_quantiles=unrolled_quantiles_tensor,
                unrolled_sample_weights=unrolled_sample_weights,
                num_output_patches=get_num_output_patches(remaining),
            )
            predictions.append(prediction)
            remaining -= prediction.shape[-1]

        batch_prediction = torch.cat(predictions, dim=-1)[..., :prediction_length].to(
            dtype=torch.float32, device="cpu"
        )

        return [batch_prediction[start:end] for (start, end) in target_idx_ranges]

    def _predict_step(
        self,
        context: torch.Tensor,
        group_ids: torch.Tensor,
        future_covariates: torch.Tensor | None,
        num_output_patches: int,
    ) -> torch.Tensor:
        kwargs = {}
        if future_covariates is not None:
            output_size = num_output_patches * self.model_output_patch_size

            if output_size > future_covariates.shape[1]:
                batch_size = len(future_covariates)
                padding_size = output_size - future_covariates.shape[1]
                padding_tensor = torch.full(
                    (batch_size, padding_size), fill_value=torch.nan, device=future_covariates.device
                )
                future_covariates = torch.cat([future_covariates, padding_tensor], dim=1)

            else:
                future_covariates = future_covariates[..., :output_size]
            kwargs["future_covariates"] = future_covariates
        with torch.no_grad():
            prediction: torch.Tensor = self.model(
                context=context, group_ids=group_ids, num_output_patches=num_output_patches, **kwargs
            ).quantile_preds.to(context)

        return prediction

    @staticmethod
    def _slide_context_and_future_covariates(
        context: torch.Tensor, future_covariates: torch.Tensor, slide_by: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # replace context with future_covariates, where the values of future covariates are known (not NaN)
        future_covariates_slice = future_covariates[..., :slide_by]
        context[..., -slide_by:] = torch.where(
            torch.isnan(future_covariates_slice), context[..., -slide_by:], future_covariates_slice
        )
        # shift future_covariates
        future_covariates = future_covariates[..., slide_by:]

        return context, future_covariates

    def predict_quantiles(  # type: ignore[override]
        self,
        inputs: TensorOrArray
        | Sequence[TensorOrArray]
        | Sequence[Mapping[str, TensorOrArray | Mapping[str, TensorOrArray]]],
        prediction_length: int | None = None,
        quantile_levels: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        **predict_kwargs,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Refer to ``Chronos2Pipeline.predict`` for shared parameters.

        Additional parameters
        ---------------------
        quantile_levels
            Quantile levels to compute, by default [0.1, 0.2, ..., 0.9]

        Returns
        -------
        quantiles
            A list of torch tensors containing quantile forecasts. Each element of the list has shape (n_variates, prediction_length, len(quantile_levels))
            and the number of elements are equal to the number of target time series (univariate or multivariate) in the `inputs`.
        mean
            A list of torch tensors containing containing mean (point) forecasts. Each element of the list has shape (n_variates, prediction_length)
            and the number of elements are equal to the number of target time series (univariate or multivariate) in the `inputs`.
        """
        training_quantile_levels = self.quantiles

        predictions: list[torch.Tensor] = self.predict(inputs, prediction_length=prediction_length, **predict_kwargs)

        # Swap quantile and time axes for each prediction
        predictions = [rearrange(pred, "... q h -> ... h q") for pred in predictions]

        if set(quantile_levels).issubset(training_quantile_levels):
            # no need to perform intra/extrapolation
            quantile_indices = [training_quantile_levels.index(q) for q in quantile_levels]
            quantiles = [pred[..., quantile_indices] for pred in predictions]
        else:
            # we interpolate quantiles if quantiles that Chronos-2 was trained on were not provided
            if min(quantile_levels) < min(training_quantile_levels) or max(quantile_levels) > max(
                training_quantile_levels
            ):
                logger.warning(
                    f"\tQuantiles to be predicted ({quantile_levels}) are not within the range of "
                    f"quantiles that Chronos-2 was trained on ({training_quantile_levels}). "
                    "Quantile predictions will be set to the minimum/maximum levels at which Chronos-2 "
                    "was trained on. This may significantly affect the quality of the predictions."
                )

            quantiles = [
                interpolate_quantiles(quantile_levels, training_quantile_levels, pred) for pred in predictions
            ]

        # NOTE: the median is returned as the mean here
        mean = [pred[..., training_quantile_levels.index(0.5)] for pred in predictions]

        return quantiles, mean

    def predict_df(
        self,
        df: "pd.DataFrame",
        future_df: "pd.DataFrame | None" = None,
        id_column: str = "item_id",
        timestamp_column: str = "timestamp",
        target: str | list[str] = "target",
        prediction_length: int | None = None,
        quantile_levels: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        batch_size: int = 256,
        **predict_kwargs,
    ) -> "pd.DataFrame":
        """
        Perform forecasting on time series data in a long-format pandas DataFrame.

        Parameters
        ----------
        df
            Time series data in long format with an id column, a timestamp, and at least one target column.
            The remaining columns in df will be treated as past-only covariates unless they are also
            present in future_df
        future_df
            Future covariates data with an id column, a timestamp, and any number of covariate columns,
            all of these columns will be treated as known future covariates
        id_column
            The name of the column which contains the unique time series identifiers, by default "item_id"
        timestamp_column
            The name of the column which contains timestamps, by default "timestamp"
            All time series in the dataframe must have regular timestamps with the same frequency (no gaps)
        target
            The name of the column(s) which contain the target variables to be forecasted, by default "target"
        prediction_length
            Number of steps to predict for each time series
        quantile_levels
            Quantile levels to compute
        batch_size
            The batch size used for prediction. Note that the batch size here means the number of time series, including target(s) and covariates,
            which are input into the model. If your data has multiple target and/or covariates, the effective number of time series tasks in a batch
            will be lower than this value, by default 256
        **predict_kwargs
            Additional arguments passed to predict_quantiles

        Returns
        -------
        The forecasts dataframe generated by the model with the following columns
        - `id_column`: The time series ID
        - `timestamp_column`: Future timestamps
        - "target_name": The name of the target column
        - "predictions": The point predictions generated by the model
        - One column for predictions at each quantile level in `quantile_levels`
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for predict_df. Please install it with `pip install pandas`.")

        if prediction_length is None:
            prediction_length = self.model_prediction_length

        if not isinstance(target, list):
            target = [target]

        inputs, original_order, prediction_timestamps = convert_df_input_to_list_of_dicts_input(
            df=df,
            future_df=future_df,
            id_column=id_column,
            timestamp_column=timestamp_column,
            target_columns=target,
            prediction_length=prediction_length,
        )

        # Generate forecasts
        quantiles, mean = self.predict_quantiles(
            inputs=inputs,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
            limit_prediction_length=False,
            batch_size=batch_size,
            **predict_kwargs,
        )
        # since predict_df tasks are homogenous by input design, we can safely stack the list of tensors into a single tensor
        quantiles_np = torch.stack(quantiles).numpy()  # [n_tasks, n_variates, horizon, num_quantiles]
        mean_np = torch.stack(mean).numpy()  # [n_tasks, n_variates, horizon]

        results_dfs = []
        for i, (series_id, future_ts) in enumerate(prediction_timestamps.items()):
            q_pred = quantiles_np[i]  # (n_variates, prediction_length, len(quantile_levels))
            point_pred = mean_np[i]  # (n_variates, prediction_length)

            for target_idx, target_col in enumerate(target):
                series_forecast_data: dict[str | tuple[str, str], Any] = {
                    id_column: series_id,
                    timestamp_column: future_ts,
                    "target_name": target_col,
                }
                series_forecast_data["predictions"] = point_pred[target_idx]
                for q_idx, q_level in enumerate(quantile_levels):
                    series_forecast_data[str(q_level)] = q_pred[target_idx, :, q_idx]

                results_dfs.append(pd.DataFrame(series_forecast_data))

        predictions_df = pd.concat(results_dfs, ignore_index=True)
        predictions_df.set_index(id_column, inplace=True)
        predictions_df = predictions_df.loc[original_order]
        predictions_df.reset_index(inplace=True)

        return predictions_df

    def _predict_fev_window(
        self,
        window: "fev.EvaluationWindow",
        quantile_levels: list[float],
        batch_size: int,
        as_univariate: bool,
        **predict_kwargs,
    ) -> tuple["datasets.DatasetDict", float]:
        import datasets
        import fev

        from chronos.chronos2.dataset import convert_fev_window_to_list_of_dicts_input

        inputs, target_columns, past_dynamic_columns, known_dynamic_columns = (
            convert_fev_window_to_list_of_dicts_input(window=window, as_univariate=as_univariate)
        )

        num_variates: int = len(target_columns) + len(past_dynamic_columns) + len(known_dynamic_columns)
        if batch_size < num_variates:
            warnings.warn(
                f"batch_size ({batch_size}) is smaller than num_variates ({num_variates}) in the task. "
                f"Setting batch_size = num_variates = num_targets + num_covariates",
                category=UserWarning,
                stacklevel=3,
            )
            batch_size = num_variates

        start_time = time.monotonic()

        quantiles, mean = self.predict_quantiles(
            inputs=inputs,
            prediction_length=window.horizon,
            quantile_levels=quantile_levels,
            limit_prediction_length=False,
            batch_size=batch_size,
            **predict_kwargs,
        )
        # since fev tasks are homogenous, we can safely stack the list of tensors into a single tensor
        quantiles_np = torch.stack(quantiles).numpy()  # [n_tasks, n_variates, horizon, num_quantiles]
        mean_np = torch.stack(mean).numpy()  # [n_tasks, n_variates, horizon]

        inference_time_s = time.monotonic() - start_time

        multivariate_forecast: dict[str, dict[str, np.ndarray]] = {variate_name: {} for variate_name in target_columns}
        # mean_np is actually the median here
        point_forecast = mean_np  # [num_items, n_variates, horizon]

        for v_idx, variate_name in enumerate(target_columns):
            multivariate_forecast[variate_name]["predictions"] = point_forecast[:, v_idx]

        for q_idx, level in enumerate(quantile_levels):
            for v_idx, variate_name in enumerate(target_columns):
                multivariate_forecast[variate_name][str(level)] = quantiles_np[:, v_idx, :, q_idx]

        predictions_dict: dict = {}
        for variate_name in target_columns:
            predictions_dict[variate_name] = datasets.Dataset.from_dict(
                {
                    k: multivariate_forecast[variate_name][k]
                    for k in ["predictions"] + [str(q) for q in quantile_levels]
                }
            )
        predictions = datasets.DatasetDict(predictions_dict)
        predictions.set_format("numpy")

        if as_univariate:
            predictions = fev.utils.combine_univariate_predictions_to_multivariate(predictions, window.target_columns)

        return predictions, inference_time_s

    def predict_fev(
        self,
        task: "fev.Task",
        batch_size: int = 256,
        as_univariate: bool = False,
        finetune_kwargs: dict | None = None,
        **kwargs,
    ) -> tuple[list["datasets.DatasetDict"], float]:
        """
        Make predictions for evaluation on a fev.Task.

        Parameters
        ----------
        task
            Benchmark task on which the evaluation should be done.
        batch_size
            Batch size used during evaluation.
        as_univariate
            If True, univariate inference is done, i.e., each target is predicted independently and
            covariates, if any, are ignored.
        finetune_kwargs
            If not None, finetuning is enabled and finetune_kwargs passed to `fit()` for finetuning
        **kwargs
            Additional keyword arguments that will be forwarded to `self.predict_quantiles`.

        Returns
        -------
        predictions
            Predictions for each window, each stored as a DatasetDict
        inference_time_s
            Total time that it took to make predictions for all windows (in seconds)
        """
        from chronos.chronos2.dataset import convert_fev_window_to_list_of_dicts_input

        try:
            import fev
        except ImportError:
            raise ImportError("fev is required for predict_fev. Please install it with `pip install fev`.")

        pipeline = self
        if finetune_kwargs is not None:
            # only fine-tune the model on the first window
            first_window = task.get_window(0)
            inputs, target_columns, past_dynamic_columns, known_dynamic_columns = (
                convert_fev_window_to_list_of_dicts_input(window=first_window, as_univariate=as_univariate)
            )

            num_variates: int = len(target_columns) + len(past_dynamic_columns) + len(known_dynamic_columns)
            if batch_size < num_variates:
                warnings.warn(
                    f"batch_size ({batch_size}) is smaller than num_variates ({num_variates}) in the task. "
                    f"Setting batch_size = num_variates = num_targets + num_covariates",
                    category=UserWarning,
                    stacklevel=2,
                )
                batch_size = num_variates

            finetune_kwargs = deepcopy(finetune_kwargs)
            finetune_kwargs["prediction_length"] = first_window.horizon
            finetune_kwargs["batch_size"] = finetune_kwargs.get("batch_size", batch_size)

            try:
                pipeline = self.fit(inputs=inputs, **finetune_kwargs)
            except Exception as e:
                msg = f"Finetuning failed with error: {e}. Continuing with the pretrained model."
                warnings.warn(msg, category=UserWarning, stacklevel=2)

        predictions_per_window = []
        inference_time_s = 0.0
        for window in task.iter_windows():
            predictions, window_inference_time_s = pipeline._predict_fev_window(
                window,
                quantile_levels=task.quantile_levels,
                batch_size=batch_size,
                as_univariate=as_univariate,
                **kwargs,
            )
            predictions_per_window.append(predictions)
            inference_time_s += window_inference_time_s

        return predictions_per_window, inference_time_s

    @torch.no_grad()
    def embed(
        self, inputs: TensorOrArray | Sequence[TensorOrArray], batch_size: int = 256, context_length: int | None = None
    ) -> tuple[list[torch.Tensor], list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Get encoder embeddings for the given time series.

        Parameters
        ----------
        inputs
            The time series to get embeddings for, can be one of:
            - A 3-dimensional `torch.Tensor` or `np.ndarray` of shape (batch, n_variates, history_length). When `n_variates > 1`, information
            will be shared among the different variates of each time series in the batch.
            - A list of `torch.Tensor` or `np.ndarray` where each element can either be 1-dimensional of shape (history_length,)
            or 2-dimensional of shape (n_variates, history_length). The history_lengths may be different across elements; left-padding
            will be applied, if needed.
        batch_size
            The batch size used for generating embeddings. Note that the batch size here means the total number of time series which are input into the model.
            If your data has multiple variates, the effective number of time series tasks in a batch will be lower than this value, by default 256
        context_length
            The maximum context length used during for inference, by default set to the model's default context length

        Returns
        -------
        embeddings
            a list of `torch.Tensor` where each element has shape (n_variates, num_patches + 2, d_model) and the number of elements are equal to the number
            of target time series (univariate or multivariate) in the `inputs`. The extra +2 is due to embeddings of the [REG] token and a masked output patch token.
        loc_scale
            a list of tuples with the mean and standard deviation of each time series.
        """
        if context_length is None:
            context_length = self.model_context_length

        if context_length > self.model_context_length:
            warnings.warn(
                f"The specified context_length {context_length} is greater than the model's default context length {self.model_context_length}. "
                f"Resetting context_length to {self.model_context_length}."
            )
            context_length = self.model_context_length

        test_dataset = Chronos2Dataset.convert_inputs(
            inputs=inputs,
            context_length=context_length,
            prediction_length=0,
            batch_size=batch_size,
            output_patch_size=self.model_output_patch_size,
            mode=DatasetMode.TEST,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=None, num_workers=1, pin_memory=True, shuffle=False, drop_last=False
        )
        all_embeds: list[torch.Tensor] = []
        all_loc_scales: list[tuple[torch.Tensor, torch.Tensor]] = []
        for batch in test_loader:
            assert batch["future_target"] is None
            batch_context = batch["context"]
            batch_group_ids = batch["group_ids"]
            batch_target_idx_ranges = batch["target_idx_ranges"]

            encoder_outputs, (locs, scales), *_ = self.model.encode(
                context=batch_context.to(device=self.model.device, dtype=torch.float32),
                group_ids=batch_group_ids.to(self.model.device),
            )
            batch_embeds = [encoder_outputs[0][start:end].cpu() for (start, end) in batch_target_idx_ranges]
            batch_loc_scales = list(
                zip(
                    [locs[start:end].cpu() for (start, end) in batch_target_idx_ranges],
                    [scales[start:end].cpu() for (start, end) in batch_target_idx_ranges],
                )
            )
            all_embeds.extend(batch_embeds)
            all_loc_scales.extend(batch_loc_scales)

        return all_embeds, all_loc_scales

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        Load the model, either from a local path, S3 prefix or from the HuggingFace Hub.
        Supports the same arguments as ``AutoConfig`` and ``AutoModel`` from ``transformers``.
        """

        if str(pretrained_model_name_or_path).startswith("s3://"):
            return BaseChronosPipeline.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        assert hasattr(config, "chronos_config"), "Not a Chronos config file"

        architecture = config.architectures[0]
        class_ = getattr(chronos.chronos2, architecture)

        if class_ is None:
            logger.warning(f"Unknown architecture: {architecture}, defaulting to Chronos2Model")
            class_ = Chronos2Model

        model = class_.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        return cls(model=model)

    def save_pretrained(self, save_directory: str | Path, *args, **kwargs):
        """
        Save the underlying model to a local directory or to HuggingFace Hub.
        """
        self.model.save_pretrained(save_directory, *args, **kwargs)
