import itertools
import numpy as np
import torch
import typer

from chronos import ChronosTokenizer
from chronosx.utils.transform import DropValues
from gluonts.dataset.field_names import FieldName
from gluonts.itertools import Cyclic
from torch.utils.data import IterableDataset, get_worker_info
from typing import List, Iterator, Optional
from gluonts.transform import (
    FilterTransformation,
    TestSplitSampler,
    ValidationSplitSampler,
    InstanceSplitter,
    ExpectedNumInstanceSampler,
    MissingValueImputation,
    Transformation,
    AddObservedValuesIndicator,
    SelectFields,
    VstackFeatures,
    Chain,
    LeavesMissingValues,
)




app = typer.Typer(pretty_exceptions_enable=False)


class PseudoShuffledIterableDataset(IterableDataset):
    """
    Shuffle entries from an iterable by temporarily accumulating them
    in an intermediate buffer.

    Parameters
    ----------
    base_dataset
        The original iterable object, representing the dataset.
    shuffle_buffer_length
        Size of the buffer use to shuffle entries from the base dataset.
    """

    def __init__(
        self, base_dataset, shuffle_buffer_length: int = 100, random_seed: int = None
    ) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.shuffle_buffer_length = shuffle_buffer_length
        self.generator = torch.Generator()
        if random_seed:
            self.generator.manual_seed(random_seed)

    def __iter__(self):
        shuffle_buffer = []

        for element in self.base_dataset:
            shuffle_buffer.append(element)
            if len(shuffle_buffer) >= self.shuffle_buffer_length:
                idx = torch.randint(
                    len(shuffle_buffer), size=(), generator=self.generator
                )
                yield shuffle_buffer.pop(idx)

        while shuffle_buffer:
            idx = torch.randint(len(shuffle_buffer), size=(), generator=self.generator)
            yield shuffle_buffer.pop(idx)


class ShuffleMixin:
    """
    Mix-in class that datasets can inherit from to get
    shuffling functionality.
    """

    def shuffle(self, shuffle_buffer_length: int = 100, random_seed: int = None):
        return PseudoShuffledIterableDataset(self, shuffle_buffer_length, random_seed)


class ChronosDataset(IterableDataset, ShuffleMixin):
    """
    Dataset wrapper, using a ``ChronosTokenizer`` to turn data from a time series
    into a HuggingFace-compatible set of ``input_ids``, ``attention_mask`` and
    ``labels``.

    Entries from the original datasets are assumed to have a ``"start"`` attribute
    (of type ``pd.Period``), and a ``"target"`` attribute (of type ``np.ndarray``).

    Parameters
    ----------
    datasets
        Datasets containing the original time series data.
    probabilities
        In training mode, data will be sampled from each of the original datasets
        with these probabilities.
    tokenizer
        Tokenizer to be used to turn sequences of real numbers into token IDs.
    context_length
        Samples context will be limited to this length.
    prediction_length
        Samples labels will be limited to this length.
    drop_prob
        In training mode, observations from a sample will be turned into ``np.nan``,
        i.e. turned into missing values, with this probability.
    min_past
        Data samples will be considered only if there's at least ``min_past``-many
        historical observations.
    mode
        One of ``"training"``, ``"validation"``, or ``"test"``.
    np_dtype
        Numpy float data type.
    """

    def __init__(
        self,
        datasets: list,
        probabilities: List[float],
        tokenizer: ChronosTokenizer,
        context_length: int = 512,
        prediction_length: int = 64,
        drop_prob: float = 0.2,
        min_past: Optional[int] = None,
        model_type: str = "seq2seq",
        imputation_method: Optional[MissingValueImputation] = None,
        mode: str = "training",
        np_dtype=np.float32,
        include_covariates: bool = True,
    ) -> None:
        super().__init__()

        assert len(probabilities) == len(datasets)
        assert mode in ("training", "validation", "test")
        assert model_type in ("seq2seq", "causal")

        self.datasets = datasets
        self.probabilities = probabilities
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.drop_prob = drop_prob if model_type == "seq2seq" else 0.0
        self.min_past = min_past or prediction_length
        self.model_type = model_type
        self.imputation_method = imputation_method or LeavesMissingValues()
        self.mode = mode
        self.np_dtype = np_dtype
        self.include_covariates = include_covariates
        self.time_series_fields = [
            FieldName.OBSERVED_VALUES,
        ]

        if self.include_covariates:
            self.time_series_fields.append(FieldName.FEAT_DYNAMIC_REAL)

    def preprocess_entry(self, entry: dict, mode: str) -> dict:
        entry = {f: entry[f] for f in ["start", "target"]}
        entry["target"] = np.asarray(entry["target"], dtype=self.np_dtype)
        assert entry["target"].ndim == 1, f"got {entry['target'].ndim=}, expected 1"

        if self.model_type == "causal":
            # Causal models do not play nice with missing values, so it is
            # recommended to use an imputation method, e.g., LastValueImputation
            entry["target"] = self.imputation_method(entry["target"])

        if mode == "training" and self.drop_prob > 0:
            target = entry["target"].copy()
            drop_p = np.random.uniform(low=0.0, high=self.drop_prob)
            mask = np.random.choice(
                [True, False], size=len(target), p=[drop_p, 1 - drop_p]
            )
            target[mask] = np.nan
            entry["target"] = target

        return entry

    # to adapt for covariates
    def create_transformation(self) -> Transformation:

        fields = [
            FieldName.START,
            FieldName.TARGET,
        ]
        if self.include_covariates:
            fields.append(FieldName.FEAT_DYNAMIC_REAL)

        return (
            SelectFields(
                fields,
                allow_missing=True,
            )
            + DropValues(drop_prob=0.2, target_field=FieldName.TARGET)
            + AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            )
        )

    def _create_instance_splitter(self, mode: str):
        assert mode in ["training", "test", "validation"]

        instance_sampler = {
            "training": ExpectedNumInstanceSampler(
                num_instances=1.0,
                min_instances=1,
                min_past=self.min_past,
                min_future=self.prediction_length,
            ),
            "test": TestSplitSampler(),
            "validation": ValidationSplitSampler(min_future=self.prediction_length),
        }[mode]

        return InstanceSplitter(
            target_field="target",
            is_pad_field="is_pad",
            start_field="start",
            forecast_start_field="forecast_start",
            instance_sampler=instance_sampler,
            past_length=self.context_length,
            future_length=self.prediction_length,
            time_series_fields=self.time_series_fields,  # to adapt for covariates
            dummy_value=np.nan,
            # dummy_value=0.0,
        )

    def create_training_data(self, data):
        data = Cyclic(data)
        split_transform = self._create_instance_splitter(
            "training"
        ) + FilterTransformation(
            condition=lambda entry: (~np.isnan(entry["past_target"])).sum() > 0
        )
        data = split_transform.apply(data, is_train=True)
        return data

    def create_test_data(self, data):
        data = self._create_instance_splitter("test").apply(data, is_train=False)
        return data

    def create_validation_data(self, data):
        data = self._create_instance_splitter("validation").apply(data, is_train=False)
        return data

    def to_hf_format(self, entry: dict) -> dict:
        past_target = torch.tensor(entry["past_target"]).unsqueeze(0)
        input_ids, attention_mask, scale = self.tokenizer.context_input_transform(
            past_target
        )

        future_target = torch.tensor(entry["future_target"]).unsqueeze(0)

        if self.mode == "test":
            labels = future_target.detach().clone()
        else:
            labels, labels_mask = self.tokenizer.label_input_transform(
                future_target, scale
            )
            labels[labels_mask == 0] = -100

        # normalize covariates by mean of absolute values
        past_covariates = entry.get("past_feat_dynamic_real", None)
        if past_covariates is not None:
            #  missing value imputation for covariates
            past_covariates[np.isnan(past_covariates)] = 0

            covariates_scale = np.mean(np.abs(past_covariates), axis=0)
            covariates_scale[covariates_scale < 1] = 1.0
            past_covariates = past_covariates / covariates_scale

        future_covariates = entry.get("future_feat_dynamic_real", None)
        if future_covariates is not None:
            #  missing value imputation for covariates
            future_covariates[np.isnan(future_covariates)] = 0

            future_covariates = future_covariates / covariates_scale

        # shift covariates by one
        if past_covariates is not None:
            num_covariates = past_covariates.shape[-1]
            past_covariates = np.concatenate(
                [past_covariates, np.array([[0] * num_covariates])], axis=0
            )

        if future_covariates is not None:
            num_covariates = future_covariates.shape[-1]
            future_covariates = np.concatenate(
                [future_covariates, np.array([[0] * num_covariates])], axis=0
            )

        if self.model_type == "causal":
            # The InstanceSplitter pads time series on the left to be equal to the
            # context_length. However, certain models (e.g., GPT2) with absolute
            # position embeddings should not be trained with left padding.
            # The following piece of code moves padding from left to right.

            assert input_ids.shape[-1] == entry["past_is_pad"].shape[0]

            # Find the index where padding starts
            pad_start_idx = np.searchsorted(1 - entry["past_is_pad"], 1)
            padded_input_ids, obs_input_ids = torch.tensor_split(
                input_ids, [pad_start_idx], dim=-1
            )
            padded_attention_mask, obs_attention_mask = torch.tensor_split(
                attention_mask, [pad_start_idx], dim=-1
            )

            # Move padding to the right
            input_ids = torch.cat(
                [
                    obs_input_ids,
                    labels,
                    padded_input_ids,
                ],
                axis=-1,
            )
            attention_mask = torch.cat(
                [
                    obs_attention_mask,
                    labels_mask,
                    padded_attention_mask,
                ],
                axis=-1,
            )

            # labels for causal models are same as the input_ids.
            # Internally transformers shifts the labels by one during training.
            labels = input_ids.clone()
            input_ids[~attention_mask] = self.tokenizer.config.pad_token_id
            labels[~attention_mask] = -100
        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
            "past_covariates": past_covariates.astype(np.float32),
            "future_covariates": future_covariates.astype(np.float32),
            "scale": scale,
        }

    def __iter__(self) -> Iterator:
        transformation = self.create_transformation()

        if self.include_covariates:
            field_time = (
                FieldName.FEAT_TIME
                if not self.include_covariates
                else FieldName.FEAT_DYNAMIC_REAL
            )
            transformation = Chain(
                [
                    transformation,
                    VstackFeatures(
                        output_field=FieldName.FEAT_DYNAMIC_REAL,
                        input_fields=[field_time],
                    ),
                ]
            )

        if self.mode == "training":
            transformed_datasets = [
                transformation.apply(dataset, is_train=True)
                for dataset in self.datasets
            ]
            iterables = [
                self.create_training_data(transformed_dataset)
                for transformed_dataset in transformed_datasets
            ]
        elif self.mode == "validation":
            transformed_datasets = [
                transformation.apply(dataset, is_train=False)
                for dataset in self.datasets
            ]
            iterables = [
                self.create_validation_data(transformed_dataset)
                for transformed_dataset in transformed_datasets
            ]
        elif self.mode == "test":
            transformed_datasets = [
                transformation.apply(dataset, is_train=False)
                for dataset in self.datasets
            ]
            iterables = [
                self.create_test_data(transformed_dataset)
                for transformed_dataset in transformed_datasets
            ]

        worker_info = get_worker_info()
        if worker_info is None:
            probs = list(self.probabilities)
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            iterables = list(itertools.islice(iterables, worker_id, None, num_workers))
            probs = list(
                itertools.islice(self.probabilities, worker_id, None, num_workers)
            )

        probs = [prob / sum(probs) for prob in probs]

        iterators = list(map(iter, iterables))
        if self.mode == "training":
            while True:
                idx = np.random.choice(range(len(iterators)), p=probs)
                try:
                    yield self.to_hf_format(next(iterators[idx]))
                except StopIteration:
                    probs[idx] = 0
                    if sum(probs) == 0:
                        return
                    probs = [prob / sum(probs) for prob in probs]
        else:
            for entry in itertools.chain(*iterators):
                yield self.to_hf_format(entry)
