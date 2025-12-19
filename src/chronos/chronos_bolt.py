# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Authors: Abdul Fatir Ansari <ansarnd@amazon.com>, Caner Turkmen <atturkm@amazon.com>, Lorenzo Stella <stellalo@amazon.com>
# Original source:
# https://github.com/autogluon/autogluon/blob/f57beb26cb769c6e0d484a6af2b89eab8aee73a8/timeseries/src/autogluon/timeseries/models/chronos/pipeline/chronos_bolt.py

import copy
import logging
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.t5.modeling_t5 import (
    ACT2FN,
    T5Config,
    T5LayerNorm,
    T5PreTrainedModel,
    T5Stack,
)
from transformers.utils import ModelOutput

from .base import BaseChronosPipeline, ForecastType


logger = logging.getLogger(__file__)


@dataclass
class ChronosBoltConfig:
    context_length: int
    prediction_length: int
    input_patch_size: int
    input_patch_stride: int
    quantiles: List[float]
    use_reg_token: bool = False


@dataclass
class ChronosBoltOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    quantile_preds: Optional[torch.Tensor] = None
    attentions: Optional[torch.Tensor] = None
    cross_attentions: Optional[torch.Tensor] = None


class Patch(nn.Module):
    def __init__(self, patch_size: int, patch_stride: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]

        if length % self.patch_size != 0:
            padding_size = (
                *x.shape[:-1],
                self.patch_size - (length % self.patch_size),
            )
            padding = torch.full(size=padding_size, fill_value=torch.nan, dtype=x.dtype, device=x.device)
            x = torch.concat((padding, x), dim=-1)

        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        return x


class InstanceNorm(nn.Module):
    """
    Apply standardization along the last dimension and optionally apply arcsinh after standardization.
    """

    def __init__(self, eps: float = 1e-5, use_arcsinh: bool = False) -> None:
        super().__init__()
        self.eps = eps
        self.use_arcsinh = use_arcsinh

    def forward(
        self, x: torch.Tensor, loc_scale: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if loc_scale is None:
            loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
            scale = torch.nan_to_num((x - loc).square().nanmean(dim=-1, keepdim=True).sqrt(), nan=1.0)
            scale = torch.where(scale == 0, self.eps, scale)
        else:
            loc, scale = loc_scale

        scaled_x = (x - loc) / scale

        if self.use_arcsinh:
            scaled_x = torch.arcsinh(scaled_x)

        return scaled_x.to(orig_dtype), (loc, scale)

    def inverse(self, x: torch.Tensor, loc_scale: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        loc, scale = loc_scale

        if self.use_arcsinh:
            x = torch.sinh(x)

        x = x * scale + loc

        return x.to(orig_dtype)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        act_fn_name: str,
        dropout_p: float = 0.0,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout_p)
        self.hidden_layer = nn.Linear(in_dim, h_dim)
        self.act = ACT2FN[act_fn_name]
        self.output_layer = nn.Linear(h_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = T5LayerNorm(out_dim)

    def forward(self, x: torch.Tensor):
        hid = self.act(self.hidden_layer(x))
        out = self.dropout(self.output_layer(hid))
        res = self.residual_layer(x)

        out = out + res

        if self.use_layer_norm:
            return self.layer_norm(out)
        return out


class ChronosBoltModelForForecasting(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [  # type: ignore
        r"input_patch_embedding\.",
        r"output_patch_embedding\.",
    ]
    _keys_to_ignore_on_load_unexpected = [r"lm_head.weight"]  # type: ignore
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]  # type: ignore

    def __init__(self, config: T5Config):
        assert hasattr(config, "chronos_config"), "Not a Chronos config file"

        super().__init__(config)
        self.model_dim = config.d_model

        self.chronos_config = ChronosBoltConfig(**config.chronos_config)

        # Only decoder_start_id (and optionally REG token)
        if self.chronos_config.use_reg_token:
            config.reg_token_id = 1

        config.vocab_size = 2 if self.chronos_config.use_reg_token else 1
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # Input patch embedding layer
        self.input_patch_embedding = ResidualBlock(
            in_dim=self.chronos_config.input_patch_size * 2,
            h_dim=config.d_ff,
            out_dim=config.d_model,
            act_fn_name=config.dense_act_fn,
            dropout_p=config.dropout_rate,
        )

        # patching layer
        self.patch = Patch(
            patch_size=self.chronos_config.input_patch_size,
            patch_stride=self.chronos_config.input_patch_stride,
        )

        # instance normalization, also referred to as "scaling" in Chronos and GluonTS
        self.instance_norm = InstanceNorm()

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        self.encoder = T5Stack(encoder_config, self.shared)

        self._init_decoder(config)

        self.num_quantiles = len(self.chronos_config.quantiles)
        quantiles = torch.tensor(self.chronos_config.quantiles, dtype=self.dtype)
        self.quantiles: torch.Tensor
        self.register_buffer("quantiles", quantiles, persistent=False)

        self.output_patch_embedding = ResidualBlock(
            in_dim=config.d_model,
            h_dim=config.d_ff,
            out_dim=self.num_quantiles * self.chronos_config.prediction_length,
            act_fn_name=config.dense_act_fn,
            dropout_p=config.dropout_rate,
        )

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def _init_weights(self, module):
        super()._init_weights(module)
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(module, (self.__class__)):
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, ResidualBlock):
            module.hidden_layer.weight.data.normal_(
                mean=0.0,
                std=factor * ((self.chronos_config.input_patch_size * 2) ** -0.5),
            )
            if hasattr(module.hidden_layer, "bias") and module.hidden_layer.bias is not None:
                module.hidden_layer.bias.data.zero_()

            module.residual_layer.weight.data.normal_(
                mean=0.0,
                std=factor * ((self.chronos_config.input_patch_size * 2) ** -0.5),
            )
            if hasattr(module.residual_layer, "bias") and module.residual_layer.bias is not None:
                module.residual_layer.bias.data.zero_()

            module.output_layer.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.output_layer, "bias") and module.output_layer.bias is not None:
                module.output_layer.bias.data.zero_()

    def encode(
        self, context: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        mask = mask.to(context.dtype) if mask is not None else torch.isnan(context).logical_not().to(context.dtype)

        batch_size, _ = context.shape
        if context.shape[-1] > self.chronos_config.context_length:
            context = context[..., -self.chronos_config.context_length :]
            mask = mask[..., -self.chronos_config.context_length :]

        # scaling
        context, loc_scale = self.instance_norm(context)

        # the scaling op above is done in 32-bit precision,
        # then the context is moved to model's dtype
        context = context.to(self.dtype)
        mask = mask.to(self.dtype)

        # patching
        patched_context = self.patch(context)
        patched_mask = torch.nan_to_num(self.patch(mask), nan=0.0)
        patched_context = torch.where(patched_mask > 0.0, patched_context, 0.0)
        # concat context and mask along patch dim
        patched_context = torch.cat([patched_context, patched_mask], dim=-1)

        # attention_mask = 1 if at least one item in the patch is observed
        attention_mask = patched_mask.sum(dim=-1) > 0  # (batch_size, patched_seq_length)

        input_embeds = self.input_patch_embedding(patched_context)

        if self.chronos_config.use_reg_token:
            # Append [REG]
            reg_input_ids = torch.full(
                (batch_size, 1),
                self.config.reg_token_id,
                device=input_embeds.device,
            )
            reg_embeds = self.shared(reg_input_ids)
            input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
            attention_mask = torch.cat(
                [
                    attention_mask.to(self.dtype),
                    torch.ones_like(reg_input_ids).to(self.dtype),
                ],
                dim=-1,
            )

        encoder_outputs = self.encoder(
            attention_mask=attention_mask,
            inputs_embeds=input_embeds,
        )

        return encoder_outputs[0], loc_scale, input_embeds, attention_mask

    def forward(
        self,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> ChronosBoltOutput:
        batch_size = context.size(0)

        hidden_states, loc_scale, input_embeds, attention_mask = self.encode(context=context, mask=mask)
        sequence_output = self.decode(input_embeds, attention_mask, hidden_states)

        quantile_preds_shape = (
            batch_size,
            self.num_quantiles,
            self.chronos_config.prediction_length,
        )
        quantile_preds = self.output_patch_embedding(sequence_output).view(*quantile_preds_shape)

        loss = None
        if target is not None:
            # normalize target
            target, _ = self.instance_norm(target, loc_scale)
            target = target.unsqueeze(1)  # type: ignore
            assert self.chronos_config.prediction_length >= target.shape[-1]

            target = target.to(quantile_preds.device)
            target_mask = (
                target_mask.unsqueeze(1).to(quantile_preds.device) if target_mask is not None else ~torch.isnan(target)
            )
            target[~target_mask] = 0.0

            # pad target and target_mask if they are shorter than model's prediction_length
            if self.chronos_config.prediction_length > target.shape[-1]:
                padding_shape = (
                    *target.shape[:-1],
                    self.chronos_config.prediction_length - target.shape[-1],
                )
                target = torch.cat([target, torch.zeros(padding_shape).to(target)], dim=-1)
                target_mask = torch.cat([target_mask, torch.zeros(padding_shape).to(target_mask)], dim=-1)

            loss = (
                2
                * torch.abs(
                    (target - quantile_preds)
                    * ((target <= quantile_preds).float() - self.quantiles.view(1, self.num_quantiles, 1))
                )
                * target_mask.float()
            )
            loss = loss.mean(dim=-2)  # Mean over prediction horizon
            loss = loss.sum(dim=-1)  # Sum over quantile levels
            loss = loss.mean()  # Mean over batch

        # Unscale predictions
        quantile_preds = self.instance_norm.inverse(
            quantile_preds.view(batch_size, -1),
            loc_scale,
        ).view(*quantile_preds_shape)

        return ChronosBoltOutput(
            loss=loss,
            quantile_preds=quantile_preds,
        )

    def _init_decoder(self, config):
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

    def decode(
        self,
        input_embeds,
        attention_mask,
        hidden_states,
        output_attentions=False,
    ):
        """
        Parameters
        ----------
        input_embeds: torch.Tensor
            Patched and embedded inputs. Shape (batch_size, patched_context_length, d_model)
        attention_mask: torch.Tensor
            Attention mask for the patched context. Shape (batch_size, patched_context_length), type: torch.int64
        hidden_states: torch.Tensor
            Hidden states returned by the encoder. Shape (batch_size, patched_context_length, d_model)

        Returns
        -------
        last_hidden_state
            Last hidden state returned by the decoder, of shape (batch_size, 1, d_model)
        """
        batch_size = input_embeds.shape[0]
        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.config.decoder_start_token_id,
            device=input_embeds.device,
        )
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            output_attentions=output_attentions,
            return_dict=True,
        )

        return decoder_outputs.last_hidden_state  # sequence_outputs, b x 1 x d_model


class ChronosBoltPipeline(BaseChronosPipeline):
    """
    Pipeline for the Chronos-Bolt model.
    
    Parameters
    ----------
    model
        ChronosBoltModelForForecasting instance containing the pretrained model.
    
    Attributes
    ----------
    model
        The underlying forecasting model
    forecast_type
        Set to ForecastType.QUANTILES indicating this pipeline produces quantiles
    default_context_length
        Default context length of 2048 time steps
    
    See Also
    --------
    ChronosPipeline : Sample-based forecasting with tokenization
    Chronos2Pipeline : Advanced forecasting with covariates support
    """
    forecast_type: ForecastType = ForecastType.QUANTILES
    default_context_length: int = 2048

    def __init__(self, model: ChronosBoltModelForForecasting):
        """
        Initialize the ChronosBoltPipeline with a pretrained model.
        
        Parameters
        ----------
        model
            ChronosBoltModelForForecasting instance containing the pretrained
            transformer model configured for quantile forecasting.
        """
        super().__init__(inner_model=model)  # type: ignore
        self.model = model

    @property
    def model_context_length(self) -> int:
        return self.model.chronos_config.context_length

    @property
    def model_prediction_length(self) -> int:
        return self.model.chronos_config.prediction_length

    @property
    def quantiles(self) -> List[float]:
        return self.model.config.chronos_config["quantiles"]

    @torch.no_grad()
    def embed(
        self, context: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Extract encoder embeddings for the given time series.
        
        This method processes the input time series through patching and instance
        normalization, then extracts encoder embeddings that can be used for
        downstream tasks like clustering, classification, or similarity search.
        
        Parameters
        ----------
        context
            Input time series. Can be a 1D tensor (single series), a list
            of 1D tensors (multiple series of varying lengths), or a 2D tensor
            where the first dimension is batch size. For 2D tensors, use
            left-padding with torch.nan to align series of different lengths.
        
        Returns
        -------
        embeddings
            Encoder embeddings with shape (batch_size, num_patches + 1, d_model),
            where num_patches is the number of patches created from the input
            time series, and the extra 1 is for the [REG] token if used by the model.
            Returned on CPU in the model's dtype.
        loc_scale
            Tuple of (location, scale) tensors used for instance normalization,
            representing the mean and standard deviation of the original time series.
            Both tensors have shape (batch_size,) and are returned on CPU.
        
        Notes
        -----
        The embeddings are extracted after patching and instance normalization
        but before the decoder. They capture the encoded representation of the
        input time series in the model's latent space.
        
        If the input context is longer than the model's context length, it will
        be automatically truncated to the most recent time steps.
        """
        context_tensor = self._prepare_and_validate_context(context=context)
        model_context_length = self.model.config.chronos_config["context_length"]

        if context_tensor.shape[-1] > model_context_length:
            context_tensor = context_tensor[..., -model_context_length:]

        context_tensor = context_tensor.to(
            device=self.model.device,
            dtype=torch.float32,
        )
        embeddings, loc_scale, *_ = self.model.encode(context=context_tensor)
        return embeddings.cpu(), (
            loc_scale[0].squeeze(-1).cpu(),
            loc_scale[1].squeeze(-1).cpu(),
        )

    def predict(
        self,
        inputs: Union[torch.Tensor, List[torch.Tensor]],
        prediction_length: Optional[int] = None,
        limit_prediction_length: bool = False,
    ) -> torch.Tensor:
        """
        Generate quantile forecasts for the given time series.
        
        This method directly predicts quantiles without generating sample trajectories.
        For predictions longer than the model's built-in horizon, it uses an
        autoregressive approach that expands the batch size by the number of quantiles
        to generate more robust long-horizon forecasts.
        
        Parameters
        ----------
        inputs
            Input time series. Can be a 1D tensor (single series), a list
            of 1D tensors (multiple series of varying lengths), or a 2D tensor
            where the first dimension is batch size. For 2D tensors, use
            left-padding with torch.nan to align series of different lengths.
        prediction_length
            Number of time steps to forecast. If not provided, uses the model's
            default prediction length from the configuration.
        limit_prediction_length
            When True, raises an error if prediction_length exceeds the model's
            built-in prediction length. When False (default), allows longer
            predictions with a warning about potential quality degradation.
        
        Returns
        -------
        torch.Tensor
            Quantile forecasts with shape (batch_size, num_quantiles, prediction_length),
            where num_quantiles is the number of quantiles the model was trained on.
            For official Chronos-Bolt models, num_quantiles is 9 for quantiles
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].
            Returned in fp32 on CPU.
        
        Raises
        ------
        ValueError
            If limit_prediction_length is True and prediction_length exceeds
            the model's built-in prediction length.
        
        Notes
        -----
        For predictions longer than the model's built-in horizon, the method uses
        an autoregressive approach:
        1. Generate initial quantiles for the first chunk
        2. Expand context by num_quantiles (treating each quantile as a scenario)
        3. Generate next chunk for each scenario
        4. Compute empirical quantiles across all scenarios
        5. Repeat until desired prediction_length is reached
        
        This approach scales the batch size by num_quantiles for long horizons,
        which may require more GPU memory but produces more robust predictions.
        
        If the input context is longer than the model's context length, it will
        be automatically truncated to the most recent time steps.
        """
        context_tensor = self._prepare_and_validate_context(context=inputs)

        if prediction_length is None:
            prediction_length = self.model_prediction_length

        if prediction_length > self.model_prediction_length:
            msg = (
                f"We recommend keeping prediction length <= {self.model_prediction_length}. "
                "The quality of longer predictions may degrade since the model is not optimized for it. "
            )
            if limit_prediction_length:
                msg += "You can turn off this check by setting `limit_prediction_length=False`."
                raise ValueError(msg)
            warnings.warn(msg)

        predictions = []
        remaining = prediction_length

        # We truncate the context here because otherwise batches with very long
        # context could take up large amounts of GPU memory unnecessarily.
        if context_tensor.shape[-1] > self.model_context_length:
            context_tensor = context_tensor[..., -self.model_context_length :]

        context_tensor = context_tensor.to(device=self.model.device, dtype=torch.float32)
        # First block prediction
        with torch.no_grad():
            prediction: torch.Tensor = self.model(context=context_tensor).quantile_preds.to(context_tensor)

            predictions.append(prediction)
            remaining -= prediction.shape[-1]

        # NOTE: The following heuristic for better prediction intervals with long-horizon forecasts
        # uses all quantiles generated by the model for the first `model_prediction_length` steps,
        # concatenating each quantile with the context and generating the next `model_prediction_length` steps.
        # The `num_quantiles * num_quantiles` "samples" thus generated are then reduced to `num_quantiles`
        # by computing empirical quantiles. Note that this option scales the batch size by `num_quantiles`
        # when the `prediction_length` is greater than `model_prediction_length`.

        if remaining > 0:
            # Expand the context along quantile axis
            context_tensor = context_tensor.unsqueeze(1).repeat(1, len(self.quantiles), 1)

        quantile_tensor = torch.tensor(self.quantiles, device=context_tensor.device)
        while remaining > 0:
            # Append the prediction to context
            context_tensor = torch.cat([context_tensor, prediction], dim=-1)[..., -self.model_context_length :]
            (batch_size, n_quantiles, context_length) = context_tensor.shape

            with torch.no_grad():
                # Reshape (batch, n_quantiles, context_length) -> (batch * n_quantiles, context_length)
                prediction = self.model(
                    context=context_tensor.reshape(batch_size * n_quantiles, context_length)
                ).quantile_preds.to(context_tensor)
            # Reshape predictions from (batch * n_quantiles, n_quantiles, model_prediction_length) to (batch, n_quantiles * n_quantiles, model_prediction_length)
            prediction = prediction.reshape(batch_size, n_quantiles * n_quantiles, -1)
            # Reduce `n_quantiles * n_quantiles` to n_quantiles and transpose back to (batch_size, n_quantiles, model_prediction_length)
            prediction = torch.quantile(prediction, q=quantile_tensor, dim=1).transpose(0, 1)

            predictions.append(prediction)
            remaining -= prediction.shape[-1]

        return torch.cat(predictions, dim=-1)[..., :prediction_length].to(dtype=torch.float32, device="cpu")

    def predict_quantiles(
        self,
        inputs: Union[torch.Tensor, List[torch.Tensor]],
        prediction_length: Optional[int] = None,
        quantile_levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        **predict_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate quantile and mean forecasts for given time series.
        
        This method generates forecasts at the specified quantile levels. If the
        requested quantiles match those the model was trained on, they are returned
        directly. Otherwise, the method performs interpolation or extrapolation
        to obtain the requested quantiles.
        
        Parameters
        ----------
        inputs
            Input time series. Can be a 1D tensor (single series), a list
            of 1D tensors (multiple series of varying lengths), or a 2D tensor
            where the first dimension is batch size. For 2D tensors, use
            left-padding with torch.nan to align series of different lengths.
        prediction_length
            Number of time steps to forecast. If not provided, uses the model's
            default prediction length from the configuration.
        quantile_levels
            List of quantile levels to compute, each between 0 and 1.
            Default is [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].
        **predict_kwargs
            Additional keyword arguments passed to the predict method, such as
            limit_prediction_length.
        
        Returns
        -------
        quantiles
            Tensor of quantile forecasts with shape
            (batch_size, prediction_length, num_quantiles).
            Returned in fp32 on CPU.
        mean
            Tensor of mean forecasts with shape (batch_size, prediction_length).
            This is actually the median (0.5 quantile) from the model's predictions.
            Returned in fp32 on CPU.
        
        Notes
        -----
        If the requested quantile_levels are a subset of the model's training
        quantiles, they are extracted directly without interpolation.
        
        If quantile_levels include values outside the range of training quantiles,
        the method will extrapolate using the minimum/maximum training quantiles,
        which may significantly affect prediction quality. A warning will be issued
        in this case.
        
        The interpolation/extrapolation assumes the model's training quantiles
        formed an equidistant grid (e.g., 0.1, 0.2, ..., 0.9), which holds for
        official Chronos-Bolt models but may not be true for custom models.
        
        The mean returned is actually the median (0.5 quantile) from the model's
        predictions, not a true mean.
        """
        # shape (batch_size, prediction_length, len(training_quantile_levels))
        predictions = (
            self.predict(inputs, prediction_length=prediction_length, **predict_kwargs).detach().swapaxes(1, 2)
        )

        training_quantile_levels = self.quantiles

        if set(quantile_levels).issubset(set(training_quantile_levels)):
            # no need to perform intra/extrapolation
            quantiles = predictions[..., [training_quantile_levels.index(q) for q in quantile_levels]]
        else:
            # we rely on torch for interpolating quantiles if quantiles that
            # Chronos Bolt was trained on were not provided
            if min(quantile_levels) < min(training_quantile_levels) or max(quantile_levels) > max(
                training_quantile_levels
            ):
                logger.warning(
                    f"\tQuantiles to be predicted ({quantile_levels}) are not within the range of "
                    f"quantiles that Chronos-Bolt was trained on ({training_quantile_levels}). "
                    "Quantile predictions will be set to the minimum/maximum levels at which Chronos-Bolt "
                    "was trained on. This may significantly affect the quality of the predictions."
                )

            # TODO: this is a hack that assumes the model's quantiles during training (training_quantile_levels)
            # made up an equidistant grid along the quantile dimension. i.e., they were (0.1, 0.2, ..., 0.9).
            # While this holds for official Chronos-Bolt models, this may not be true in the future, and this
            # function may have to be revised.
            augmented_predictions = torch.cat(
                [predictions[..., [0]], predictions, predictions[..., [-1]]],
                dim=-1,
            )
            quantiles = torch.quantile(
                augmented_predictions,
                q=torch.tensor(quantile_levels, dtype=augmented_predictions.dtype),
                dim=-1,
            ).permute(1, 2, 0)
        # NOTE: the median is returned as the mean here
        mean = predictions[:, :, training_quantile_levels.index(0.5)]
        return quantiles, mean

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        Load a pretrained ChronosBoltPipeline from various sources.
        
        This method loads a pretrained ChronosBoltPipeline model from a local path,
        S3 bucket, or the HuggingFace Hub. It automatically instantiates the
        appropriate model architecture based on the configuration.
        
        Parameters
        ----------
        pretrained_model_name_or_path
            Path or identifier for the pretrained model. Can be:
            - A local directory path containing model files
            - An S3 URI (s3://bucket/prefix)
            - A HuggingFace Hub model identifier (e.g., "amazon/chronos-bolt-small")
        *args
            Additional positional arguments passed to AutoConfig and the model constructor.
        **kwargs
            Additional keyword arguments passed to AutoConfig and the model constructor.
            Common options include:
            - torch_dtype: Data type for model weights ("auto", "float32", "bfloat16")
            - device_map: Device placement strategy for model layers
            - Other transformers AutoConfig and model arguments
        
        Returns
        -------
        ChronosBoltPipeline
            An instance of ChronosBoltPipeline with the loaded model.
        
        Raises
        ------
        AssertionError
            If the configuration is not a valid Chronos config.
        
        Notes
        -----
        For S3 URIs, the method delegates to BaseChronosPipeline.from_pretrained
        which handles S3 download and caching.
        
        The method automatically detects the model architecture from the configuration
        and instantiates the appropriate class. If the architecture is not recognized,
        it defaults to ChronosBoltModelForForecasting.
        
        This method supports all arguments accepted by HuggingFace's AutoConfig
        and model classes.
        """

        if str(pretrained_model_name_or_path).startswith("s3://"):
            return BaseChronosPipeline.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        assert hasattr(config, "chronos_config"), "Not a Chronos config file"

        architecture = config.architectures[0]
        class_ = globals().get(architecture)

        if class_ is None:
            logger.warning(f"Unknown architecture: {architecture}, defaulting to ChronosBoltModelForForecasting")
            class_ = ChronosBoltModelForForecasting

        model = class_.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        return cls(model=model)
