# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_map, tree_unflatten
from transformers import T5Config

from chronos.mlx.t5 import T5
from chronos.mlx.translate import translate_weights


@dataclass
class ChronosConfig:
    """
    This class holds all the configuration parameters to be used
    by ``ChronosTokenizer`` and ``ChronosModel``.
    """

    tokenizer_class: str
    tokenizer_kwargs: Dict[str, Any]
    n_tokens: int
    n_special_tokens: int
    pad_token_id: int
    eos_token_id: int
    use_eos_token: bool
    model_type: Literal["causal", "seq2seq"]
    context_length: int
    prediction_length: int
    num_samples: int
    temperature: float
    top_k: int
    top_p: float

    def __post_init__(self):
        assert (
            self.pad_token_id < self.n_special_tokens
            and self.eos_token_id < self.n_special_tokens
        ), f"Special token id's must be smaller than {self.n_special_tokens=}"

    def create_tokenizer(self) -> "ChronosTokenizer":
        if self.tokenizer_class == "MeanScaleUniformBins":
            return MeanScaleUniformBins(**self.tokenizer_kwargs, config=self)
        raise ValueError


class ChronosTokenizer:
    """
    A ``ChronosTokenizer`` defines how time series are mapped into token IDs
    and back.

    For details, see the ``input_transform`` and ``output_transform`` methods,
    which concrete classes must implement.
    """

    def input_transform(
        self, context: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Any]:
        """
        Turn a batch of time series into token IDs, attention map, and scale.

        Parameters
        ----------
        context
            A numpy array shaped (batch_size, time_length), containing the
            timeseries to forecast. Use left-padding with ``np.nan``
            to align time series of different lengths.

        Returns
        -------
        token_ids
            A numpy array of integers, shaped (batch_size, time_length + 1)
            if ``config.use_eos_token`` and (batch_size, time_length)
            otherwise, containing token IDs for the input series.
        attention_mask
            A boolean numpy array, same shape as ``token_ids``, indicating
            which input observations are not ``np.nan`` (i.e. not
            missing nor padding).
        tokenizer_state
            An object that will be passed to ``output_transform``.
            Contains the relevant context to decode output samples into
            real values, such as location and scale parameters.
        """
        raise NotImplementedError()

    def output_transform(self, samples: np.ndarray, tokenizer_state: Any) -> np.ndarray:
        """
        Turn a batch of sample token IDs into real values.

        Parameters
        ----------
        samples
            A numpy array of integers, shaped (batch_size, num_samples, time_length),
            containing token IDs of sample trajectories.
        tokenizer_state
            An object returned by ``input_transform`` containing
            relevant context to decode samples, such as location and scale.
            The nature of this depends on the specific tokenizer.

        Returns
        -------
        forecasts
            A real numpy array, shaped (batch_size, num_samples, time_length),
            containing forecasted sample paths.
        """
        raise NotImplementedError()


class MeanScaleUniformBins(ChronosTokenizer):
    def __init__(
        self, low_limit: float, high_limit: float, config: ChronosConfig
    ) -> None:
        self.config = config
        self.centers = np.linspace(
            low_limit,
            high_limit,
            config.n_tokens - config.n_special_tokens - 1,
        )
        self.boundaries = np.concatenate(
            (
                np.array([-1e20]),
                (self.centers[1:] + self.centers[:-1]) / 2,
                np.array([1e20]),
            )
        )

    def input_transform(
        self, context: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch_size, length = context.shape

        if length > self.config.context_length:
            context = context[..., -self.config.context_length :]

        attention_mask = ~np.isnan(context)
        scale = np.nansum(np.abs(context) * attention_mask, axis=-1) / np.nansum(
            attention_mask, axis=-1
        )
        scale[~(scale > 0)] = 1.0
        scaled_context = context / scale[..., np.newaxis]
        token_ids = (
            np.digitize(scaled_context, bins=self.boundaries)
            + self.config.n_special_tokens
        )
        token_ids[~attention_mask] = self.config.pad_token_id

        if self.config.use_eos_token:
            eos_tokens = np.full((batch_size, 1), fill_value=self.config.eos_token_id)
            token_ids = np.concatenate((token_ids, eos_tokens), axis=1)
            eos_mask = np.full((batch_size, 1), fill_value=True)
            attention_mask = np.concatenate((attention_mask, eos_mask), axis=1)

        return token_ids, attention_mask, scale

    def output_transform(self, samples: np.ndarray, scale: np.ndarray) -> np.ndarray:
        scale_unsqueezed = scale[..., np.newaxis, np.newaxis]
        indices = np.clip(
            samples - self.config.n_special_tokens,
            a_min=0,
            a_max=len(self.centers) - 1,
        )
        return self.centers[indices] * scale_unsqueezed


class ChronosModel(nn.Module):
    """
    A ``ChronosModel`` wraps a ``T5`` object from ``chronos.mlx.t5``
    and uses it to predict sample paths for time series tokens.

    Parameters
    ----------
    config
        The configuration to use.
    model
        The pretrained T5 model to use.
    """

    def __init__(self, config: ChronosConfig, model: T5) -> None:
        super().__init__()
        assert config.model_type == "seq2seq" and isinstance(
            model, T5
        ), "Only the T5 model is currently supported in MLX"
        self.config = config
        self.model = model

    def encode(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
    ):
        """
        Extract the encoder embedding for the given token sequences.

        Parameters
        ----------
        input_ids
            Array of indices of input sequence tokens in the vocabulary
            with shape (batch_size, sequence_length).
        attention_mask
            A mask array of the same shape as input_ids to avoid attending
            on padding or missing tokens.

        Returns
        -------
        embedding
            An array of encoder embeddings with shape
            (batch_size, sequence_length, d_model).
        """
        assert (
            self.config.model_type == "seq2seq"
        ), "Encoder embeddings are only supported for encoder-decoder models"
        return self.model.encode(inputs=input_ids, mask=attention_mask)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
        prediction_length: Optional[int] = None,
        num_samples: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> mx.array:
        """
        Predict future sample tokens for the given token sequences.

        Arguments ``prediction_length``, ``num_samples``, ``temperature``,
        ``top_k``, ``top_p`` can be used to customize the model inference,
        and default to the corresponding attributes in ``self.config`` if
        not provided.

        Returns
        -------
        samples
            A numpy array of integers, shaped (batch_size, num_samples, time_length),
            containing forecasted sample paths.
        """
        if prediction_length is None:
            prediction_length = self.config.prediction_length
        if num_samples is None:
            num_samples = self.config.num_samples
        if temperature is None:
            temperature = self.config.temperature
        if top_k is None:
            top_k = self.config.top_k
        if top_p is None:
            top_p = self.config.top_p

        preds = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            min_new_tokens=prediction_length,
            max_new_tokens=prediction_length,
            do_sample=True,
            num_return_sequences=num_samples,
            eos_token_id=self.config.eos_token_id,
            pad_token_id=self.config.pad_token_id,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        preds = preds[..., 1:]  # remove the decoder start token

        return preds.reshape(input_ids.shape[0], num_samples, -1)


def left_pad_and_stack_1D(arrays: List[np.ndarray]):
    max_len = max(len(c) for c in arrays)
    padded = []
    for c in arrays:
        assert isinstance(c, np.ndarray)
        assert c.ndim == 1
        padding = np.full(shape=(max_len - len(c),), fill_value=np.nan)
        padded.append(np.concatenate((padding, c), axis=-1))
    return np.stack(padded)


class ChronosPipeline:
    """
    A ``ChronosPipeline`` uses the given tokenizer and model to forecast
    input time series.

    Use the ``from_pretrained`` class method to load serialized models.
    Use the ``predict`` method to get forecasts.

    Parameters
    ----------
    tokenizer
        The tokenizer object to use.
    model
        The model to use.
    """

    tokenizer: ChronosTokenizer
    model: ChronosModel

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def _prepare_and_validate_context(
        self, context: Union[np.ndarray, List[np.ndarray]]
    ) -> np.ndarray:
        if isinstance(context, list):
            context = left_pad_and_stack_1D(context)
        assert isinstance(context, np.ndarray)
        if context.ndim == 1:
            context = context[np.newaxis, ...]
        assert context.ndim == 2

        return context

    def embed(
        self, context: Union[np.ndarray, List[np.ndarray]]
    ) -> Tuple[np.ndarray, Any]:
        """
        Get encoder embeddings for the given time series.

        Parameters
        ----------
        context
            Input series. This is either a 1D tensor, or a list
            of 1D tensors, or a 2D tensor whose first dimension
            is batch. In the latter case, use left-padding with
            ``torch.nan`` to align series of different lengths.

        Returns
        -------
        embeddings, tokenizer_state
            A tuple of two tensors: the encoder embeddings and the tokenizer_state,
            e.g., the scale of the time series in the case of mean scaling.
            The encoder embeddings are shaped (batch_size, context_length, d_model)
            or (batch_size, context_length + 1, d_model), where context_length
            is the size of the context along the time axis if a 2D tensor was provided
            or the length of the longest time series, if a list of 1D tensors was
            provided, and the extra 1 is for EOS.
        """
        context_array = self._prepare_and_validate_context(context=context)
        token_ids, attention_mask, tokenizer_state = self.tokenizer.input_transform(
            context_array
        )
        embeddings = self.model.encode(
            input_ids=mx.array(token_ids),
            attention_mask=mx.array(attention_mask),
        )
        return np.array(embeddings.astype(mx.float32)), tokenizer_state

    def predict(
        self,
        context: Union[np.ndarray, List[np.ndarray]],
        prediction_length: Optional[int] = None,
        num_samples: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        limit_prediction_length: bool = True,
    ) -> np.ndarray:
        """
        Get forecasts for the given time series.

        Parameters
        ----------
        context
            Input series. This is either a 1D numpy array, or a list
            of 1D numpy arrays, or a 2D numpy array whose first dimension
            is batch. In the latter case, use left-padding with
            ``np.nan`` to align series of different lengths.
        prediction_length
            Time steps to predict. Defaults to what specified
            in ``self.model.config``.
        num_samples
            Number of sample paths to predict. Defaults to what
            specified in ``self.model.config``.
        temperature
            Temperature to use for generating sample tokens.
            Defaults to what specified in ``self.model.config``.
        top_k
            Top-k parameter to use for generating sample tokens.
            Defaults to what specified in ``self.model.config``.
        top_p
            Top-p parameter to use for generating sample tokens.
            Defaults to what specified in ``self.model.config``.
        limit_prediction_length
            Force prediction length smaller or equal than the
            built-in prediction length from the model. True by
            default. When true, fail loudly if longer predictions
            are requested, otherwise longer predictions are allowed.

        Returns
        -------
        samples
            Numpy array of sample forecasts, of shape
            (batch_size, num_samples, prediction_length).
        """
        context_array = self._prepare_and_validate_context(context=context)

        if prediction_length is None:
            prediction_length = self.model.config.prediction_length

        if prediction_length > self.model.config.prediction_length:
            msg = (
                f"We recommend keeping prediction length <= {self.model.config.prediction_length}. "
                f"The quality of longer predictions may degrade since the model is not optimized for it. "
            )
            if limit_prediction_length:
                msg += "You can turn off this check by setting `limit_prediction_length=False`."
                raise ValueError(msg)
            warnings.warn(msg)

        predictions = []
        remaining = prediction_length

        while remaining > 0:
            token_ids, attention_mask, scale = self.tokenizer.input_transform(
                context_array
            )
            token_ids, attention_mask = mx.array(token_ids), mx.array(attention_mask)
            samples = self.model(
                token_ids,
                attention_mask,
                min(remaining, self.model.config.prediction_length),
                num_samples,
                temperature,
                top_k,
                top_p,
            )
            prediction = self.tokenizer.output_transform(np.array(samples), scale)

            predictions.append(prediction)
            remaining -= prediction.shape[-1]

            if remaining <= 0:
                break

            context_array = np.concatenate(
                [context_array, np.median(prediction, axis=1)], axis=-1
            )

        return np.concatenate(predictions, axis=-1)

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: Union[str, Path], dtype: str = "float32"
    ):
        """
        Load the model, either from a local path or from the HuggingFace Hub.

        Parameters
        ----------
        model_name_or_path
            Model ID on HuggingFace Hub or local path.
        dtype, optional
            String denoting the float dtype of the mlx model,
            by default "float32"

        Returns
        -------
            A ChronosPipeline
        """

        config = T5Config.from_pretrained(model_name_or_path)

        assert hasattr(config, "chronos_config"), "Not a Chronos config file"

        dtype = getattr(mx, dtype)
        chronos_config = ChronosConfig(**config.chronos_config)
        inner_model = T5(config=config)
        weights = translate_weights(model_name_or_path=model_name_or_path, dtype=dtype)
        weights = tree_unflatten(list(weights.items()))
        weights = tree_map(lambda p: p.astype(dtype), weights)
        inner_model.update(weights)
        mx.eval(inner_model.parameters())

        return cls(
            tokenizer=chronos_config.create_tokenizer(),
            model=ChronosModel(config=chronos_config, model=inner_model),
        )
