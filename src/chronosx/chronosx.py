import inspect
import logging
import numpy as np
import torch
import itertools

from chronos import BaseChronosPipeline
from chronos import ChronosConfig
from chronos import ChronosPipeline
from pathlib import Path
from tqdm.auto import tqdm
from transformers.generation.configuration_utils import GenerationConfig
from transformers.utils.import_utils import is_accelerate_available
from transformers import (
    AutoConfig,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from typing import Any, Dict, Optional

from gluonts.itertools import batcher
from gluonts.model.forecast import SampleForecast
from gluonts.transform import (
    InstanceSplitter,
    TestSplitSampler,
)

from chronosx.utils.transform import create_transformation

if is_accelerate_available():
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

from chronosx.injection_blocks.block_mapping import injection_blocks_map
from chronosx.utils.utils import count_parameters, compute_metrics, log_on_main
from chronosx.utils.prepare_covariates import prepare_covariates

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class ChronosX(T5ForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        T5ForConditionalGeneration.__init__(self, *args, **kwargs)

        self.output_hidden_states = False

        if self.input_injection_class:
            self.input_injection_block = self.input_injection_class(
                hidden_dim=self.hidden_dim,
                model_dim=self.model_dim,
                num_covariates=self.num_covariates,
                num_layers=self.num_layers,
            )
            self.input_injection_block_decoder = self.input_injection_class(
                hidden_dim=self.hidden_dim,
                model_dim=self.model_dim,
                num_covariates=self.num_covariates,
                num_layers=self.num_layers,
            )
        else:
            self.input_injection_block = None
            self.input_injection_block_decoder = None

        if self.output_injection_class:
            self.output_injection_block = self.output_injection_class(
                hidden_dim=self.hidden_dim,
                model_dim=self.model_dim,
                num_covariates=self.num_covariates,
                num_layers=self.num_layers,
                vocab_size=self.vocab_size,
            )
            self.output_hidden_states = True
        else:
            self.output_injection_block = None

    @classmethod
    def set_state(
        cls,
        num_covariates,
        covariate_injection,
        hidden_dim,
        num_layers,
        vocab_size,
        model_dim,
    ):

        cls.num_covariates = num_covariates
        cls.covariate_injection = covariate_injection
        cls.hidden_dim = hidden_dim
        cls.num_layers = num_layers
        cls.vocab_size = vocab_size
        cls.model_dim = model_dim

        if cls.covariate_injection in injection_blocks_map.keys():
            (cls.input_injection_class, cls.output_injection_class) = (
                injection_blocks_map[cls.covariate_injection]
            )
        else:
            cls.input_injection_class = None
            cls.output_injection_class = None

        return cls

    def initialize_blocks(self):
        if self.input_injection_block:
            self.input_injection_block.initialize_modules()
            self.input_injection_block_decoder.initialize_modules()

        if self.output_injection_block:
            self.output_injection_block.initialize_modules()

    def freeze(self, layer_name=None):
        for name, param in self.named_parameters():
            if layer_name == "all" or layer_name in name:
                param.requires_grad = False

    def unfreeze(self, layer_name=None):
        for name, param in self.named_parameters():
            if layer_name == "all" or layer_name in name:
                param.requires_grad = True

    def _inject_at_input(
        self,
        input_ids: torch.Tensor = None,
        decoder_input_ids: torch.Tensor = None,
        past_covariates: torch.Tensor = None,
        future_covariates: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        # input injection
        inputs_embeds = None
        decoder_inputs_embeds = None
        if input_ids is not None:
            inputs_embeds = self.encoder.embed_tokens(input_ids)
            inputs_embeds = self.input_injection_block(inputs_embeds, past_covariates)
            input_ids = None

        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self._shift_right(labels)

        if future_covariates is not None and decoder_input_ids is not None:
            decoder_inputs_embeds = self.decoder.embed_tokens(decoder_input_ids)

            # shifting covariates
            shifted_future_covariates = self._shift_right(
                future_covariates.transpose(1, 2)
            ).transpose(1, 2)

            decoder_inputs_embeds = self.input_injection_block(
                decoder_inputs_embeds, shifted_future_covariates, is_decoder=True
            )
            decoder_input_ids = None

        return (input_ids, decoder_input_ids, inputs_embeds, decoder_inputs_embeds)

    def _inject_at_output(self, output, labels, future_covariates):
        last_hidden_state = output.decoder_hidden_states[-1]
        if future_covariates is not None:
            output.logits, output.loss = self.output_injection_block(
                future_covariates=future_covariates,
                labels=labels,
                logits=output.logits,
                last_hidden_state=last_hidden_state,
            )

        return output

    def forward(
        self,
        input_ids: torch.Tensor = None,
        decoder_input_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        decoder_inputs_embeds: torch.Tensor = None,
        past_covariates: torch.Tensor = None,
        future_covariates: torch.Tensor = None,
        labels: torch.Tensor = None,
        **kwargs,
    ):

        if self.input_injection_block is not None:
            (input_ids, decoder_input_ids, inputs_embeds, decoder_inputs_embeds) = (
                self._inject_at_input(
                    input_ids=input_ids,
                    decoder_input_ids=decoder_input_ids,
                    past_covariates=past_covariates,
                    future_covariates=future_covariates,
                    labels=labels,
                )
            )

        # Removing key 'num_items_in_batch' from kwargs.
        # This is necessary as recent versions of transformers make the code break with it.
        kwargs_filtered = kwargs.copy()
        if "num_items_in_batch" in kwargs_filtered:
            kwargs_filtered.pop("num_items_in_batch")

        output = super(ChronosX, self).forward(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            output_hidden_states=self.output_hidden_states,
            **kwargs_filtered,
        )

        if self.output_injection_block is not None:
            output = self._inject_at_output(
                output=output, labels=labels, future_covariates=future_covariates
            )

        return output

    def generate(self, **kwargs):
        if self.output_injection_block is not None:
            self.output_injection_block.restart_generator_counter()

        if self.input_injection_block is not None:
            self.input_injection_block.restart_generator_counter()
            self.input_injection_block_decoder.restart_generator_counter()

        output = super(ChronosX, self).generate(**kwargs)

        if self.output_injection_block is not None:
            self.output_injection_block.generating = False

        if self.input_injection_block is not None:
            self.input_injection_block.generating = False
            self.input_injection_block_decoder.generating = False

        return output

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        future_covariates=None,
        **kwargs,
    ):

        kwargs_augmented = {
            **kwargs,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
            "encoder_outputs": encoder_outputs,
        }

        output = super(ChronosX, self).prepare_inputs_for_generation(
            input_ids,
            **kwargs_augmented,
        )

        output.update({"future_covariates": future_covariates})

        return output

    def _prepare_encoder_decoder_kwargs_for_generation(
        self,
        inputs_tensor: torch.Tensor,
        model_kwargs,
        model_input_name: Optional[str],
        generation_config: GenerationConfig,
    ) -> Dict[str, Any]:
        # 1. get encoder
        encoder = self.get_encoder()
        # Compatibility with Accelerate big model inference: we need the encoder to outputs stuff on the same device
        # as the inputs.
        if hasattr(self, "hf_device_map"):
            if hasattr(encoder, "_hf_hook"):
                encoder._hf_hook.io_same_device = True
            else:
                add_hook_to_module(encoder, AlignDevicesHook(io_same_device=True))

        # 2. Prepare encoder args and encoder kwargs from model kwargs and generation config.
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_accepts_wildcard = (
            "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        )
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value
                for argument, value in encoder_kwargs.items()
                if argument in encoder_signature
            }
        encoder_kwargs["output_attentions"] = generation_config.output_attentions
        encoder_kwargs["output_hidden_states"] = generation_config.output_hidden_states

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = (
            model_input_name if model_input_name is not None else self.main_input_name
        )
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)  # type: ignore

        # IMPORTANT: here is the place where we provide the updated token embeddings for Input Injection Block
        # Remember that we inject covariates on the token embeddings. This means that
        # we do not have to give to the encoder the input_ids but rather the updated token embeddings.
        # That's why we remove input_ids from *encoder_kwargs* and rather use 'inputs_embeds'
        if self.input_injection_block is not None:
            inputs_embeds = self.encoder.embed_tokens(encoder_kwargs["input_ids"])
            inputs_embeds = self.input_injection_block(
                inputs_embeds, model_kwargs["past_covariates"]
            )
            encoder_kwargs.pop("input_ids")
            encoder_kwargs["inputs_embeds"] = inputs_embeds
            model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)  # type: ignore

        return model_kwargs


class ChronosXPipeline(ChronosPipeline):
    def __init__(
        self,
        prediction_length: int,
        num_covariates: int,
        covariate_injection: str = "IIB+OIB",
        device_map: str = "cuda",  # use "cpu" for CPU inference
        hidden_dim: int = 256,
        num_layers: int = 1,
        pretrained_model_name_or_path="amazon/chronos-t5-small",
        layers_to_unfreeze: str = "injection_block",
    ):

        pretrained_kwargs = {"device_map": device_map}

        pipeline = BaseChronosPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            **pretrained_kwargs,
        )

        chronos_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            **pretrained_kwargs,
        )

        self.prediction_length = prediction_length
        self.num_covariates = num_covariates
        self.covariate_injection = covariate_injection
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.pretrained_model = pipeline.model
        self.pretrained_model_tokenizer = pipeline.tokenizer
        self.tokenizer = self.create_tokenizer()
        self.vocab_size = chronos_config.vocab_size
        self.model_dim = chronos_config.d_model
        self.layers_to_unfreeze = layers_to_unfreeze
        self.chronosx = ChronosX.set_state(
            num_covariates=self.num_covariates,
            covariate_injection=self.covariate_injection,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            vocab_size=self.vocab_size,
            model_dim=self.model_dim,
        ).from_pretrained(
            self.pretrained_model_name_or_path,
            **pretrained_kwargs,
        )

    def create_tokenizer(self):
        inputs = self.pretrained_model_tokenizer.config.__dict__
        inputs.update({"prediction_length": self.prediction_length})
        chronos_config = ChronosConfig(**inputs)
        return chronos_config.create_tokenizer()

    def prepare_model_for_finetuning(self):

        self.chronosx.initialize_blocks()

        self.chronosx.config.pad_token_id = (
            self.chronosx.generation_config.pad_token_id
        ) = 0

        self.chronosx.config.eos_token_id = (
            self.chronosx.generation_config.eos_token_id
        ) = 1

        if self.layers_to_unfreeze is not None:
            self.chronosx.freeze(layer_name="all")
            self.chronosx.unfreeze(layer_name=self.layers_to_unfreeze)

    def load_pretrained_zero_shot_model(self, device_map: str = "cuda"):
        return ChronosX.set_state(
            covariate_injection=None,
            num_covariates=self.num_covariates,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            vocab_size=self.vocab_size,
            model_dim=self.model_dim,
        ).from_pretrained(
            self.pretrained_model_name_or_path,
            **{"device_map": device_map},
        )

    def evaluate_model_on_validation_set(
        self,
        quantized_val_dataset,
        output_dir=Path(__file__).parent / "output" / "group0" / "finetune",
        dataloader_num_workers=1,
        tf32=False,
        torch_compile=0,
        per_device_eval_batch_size=8,
        eval_accumulation_steps=4,
        device_map: str = "cuda",
        covariate_injection: str = None,
    ):

        output_dir.mkdir(exist_ok=True, parents=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            dataloader_num_workers=dataloader_num_workers,
            tf32=tf32,  # remove this if not using Ampere GPUs (e.g., A100)
            torch_compile=torch_compile,
            per_device_eval_batch_size=per_device_eval_batch_size,
            eval_accumulation_steps=eval_accumulation_steps,
        )

        pretrained_model_zeroshot = ChronosX.set_state(
            covariate_injection=covariate_injection,
            num_covariates=self.num_covariates,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            vocab_size=self.vocab_size,
            model_dim=self.model_dim,
        ).from_pretrained(
            self.pretrained_model_name_or_path,
            **{"device_map": device_map},
        )

        # Create Trainer instance
        self.pretrained_model.eval()
        trainer = Trainer(
            model=pretrained_model_zeroshot,
            args=training_args,
            eval_dataset=quantized_val_dataset,
            compute_metrics=compute_metrics,
        )

        valiation_loss = trainer.evaluate(quantized_val_dataset)
        self.pretrained_model.train()

        return valiation_loss["eval_loss"]

    def train(
        self,
        output_dir=Path(__file__).parent / "output" / "group0" / "finetune",
        per_device_train_batch_size=32,
        learning_rate=0.01,
        lr_scheduler_type="linear",
        warmup_ratio=0.0,
        optim="adamw_torch_fused",
        log_steps=20,
        save_steps=100,
        max_steps=5000,
        gradient_accumulation_steps=2,
        dataloader_num_workers=1,
        tf32=False,
        torch_compile=0,
        eval_steps=100,
        per_device_eval_batch_size=8,
        eval_accumulation_steps=4,
        load_best_model_at_end=True,
        save_total_limit=5,
        quantized_train_dataset=None,
        quantized_val_dataset=None,
        seed=None,
    ):

        output_dir.mkdir(exist_ok=True, parents=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            optim=optim,
            logging_dir=str(output_dir / "logs"),
            logging_strategy="steps",
            logging_steps=log_steps,
            save_strategy="steps",
            save_steps=save_steps,
            report_to="tensorboard",
            max_steps=max_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            dataloader_num_workers=dataloader_num_workers,
            tf32=tf32,  # remove this if not using Ampere GPUs (e.g., A100)
            torch_compile=torch_compile,
            ddp_find_unused_parameters=False,
            metric_for_best_model="val_loss",
            evaluation_strategy="steps",
            eval_steps=eval_steps,
            per_device_eval_batch_size=per_device_eval_batch_size,
            eval_accumulation_steps=eval_accumulation_steps,
            load_best_model_at_end=load_best_model_at_end,
            logging_first_step=True,
            greater_is_better=False,
            save_total_limit=save_total_limit,
            # seed=seed,
        )

        # Create Trainer instance
        if quantized_val_dataset is None:
            training_args.eval_strategy = "no"

        trainer = Trainer(
            model=self.chronosx,
            args=training_args,
            train_dataset=quantized_train_dataset,
            eval_dataset=(
                {"val": quantized_val_dataset} if quantized_val_dataset else None
            ),
            compute_metrics=compute_metrics,
        )

        # prepare training for finetuning
        trainer._signature_columns = [
            "labels",
            "attention_mask",
            "input_ids",
            "past_covariates",
            "future_covariates",
            "decoder_input_ids",
        ]

        log_on_main("Training", logger)

        self.prepare_model_for_finetuning()

        parameter_count = count_parameters(self.chronosx)
        print(f"parameter_count: {parameter_count}")

        print("Model parameters:")
        for name, params in self.chronosx.named_parameters():
            print(name, params.requires_grad)

        trainer.train()

        if quantized_val_dataset:
            val_loss_finetuned_model = trainer.evaluate(quantized_val_dataset)
        else:
            val_loss_finetuned_model = {"eval_loss": np.nan}

        pretrained_model_path = output_dir / "final-checkpoint"
        self.chronosx.save_pretrained(pretrained_model_path)

        return val_loss_finetuned_model["eval_loss"]

    def predict(
        self, context: list[torch.Tensor], covariates: list[dict], num_samples: int = 20
    ):
        context_tensor = self._prepare_and_validate_context(context=context)
        token_ids, attention_mask, scale = self.tokenizer.context_input_transform(
            context_tensor
        )

        prepared_covariates = [prepare_covariates(entry) for entry in covariates]
        future_covariates = torch.tensor(
            [entry["future_covariates"] for entry in prepared_covariates]
        )
        past_covariates = torch.tensor(
            [entry["past_covariates"] for entry in prepared_covariates]
        )

        preds = self.chronosx.generate(
            input_ids=token_ids.to(self.chronosx.device),
            attention_mask=attention_mask.to(self.chronosx.device),
            generation_config=GenerationConfig(
                min_new_tokens=self.prediction_length,
                max_new_tokens=self.prediction_length,
                do_sample=True,
                num_return_sequences=num_samples,
                eos_token_id=self.chronosx.config.eos_token_id,
                pad_token_id=self.chronosx.config.pad_token_id,
            ),
            future_covariates=future_covariates.to(self.chronosx.device),
            past_covariates=past_covariates.to(self.chronosx.device),
        )

        preds = preds[..., 1:]  # remove the decoder start token
        preds = preds.reshape(token_ids.size(0), num_samples, -1)
        preds = self.tokenizer.output_transform(preds.to("cpu"), scale.to("cpu"))

        return preds.to(dtype=torch.float32, device="cpu")

    def finetune(
        self,
        output_dir,
        quantized_train_dataset,
        lr=0.01,
        quantized_val_dataset=None,
        skip_pretrained_validation=False,
        max_steps=20,
        seed=None,
    ):

        if not skip_pretrained_validation:
            assert quantized_val_dataset is not None
            val_loss_pretrained_model = self.evaluate_model_on_validation_set(
                covariate_injection=None,
                quantized_val_dataset=quantized_val_dataset,
                output_dir=output_dir,
            )
        else:
            print(f"val_loss_pretrained_model is set up to nan since skip_pretrained_validation is given as False")
            val_loss_pretrained_model = np.nan

        eval_loss_finetuned_model = self.train(
            learning_rate=lr,
            quantized_train_dataset=quantized_train_dataset,
            quantized_val_dataset=quantized_val_dataset,
            max_steps=max_steps,
            output_dir=output_dir,
            seed=seed,
        )

        if quantized_val_dataset is None:
            print(f"No Validation set is provided")

        print(
            f"lr:{lr} - val_loss_pretrained_model: {val_loss_pretrained_model:.4f} - eval_loss_finetuned_model:{eval_loss_finetuned_model:.4f}"
        )

        if val_loss_pretrained_model < eval_loss_finetuned_model:
            model = self.load_pretrained_zero_shot_model()
            val_loss = val_loss_pretrained_model
        else:
            model = self.chronosx
            val_loss = eval_loss_finetuned_model

        save_model_path = output_dir / "final-checkpoint"
        model.save_pretrained(save_model_path)

        return val_loss, save_model_path


    def generate_forecasts(
        self,
        test_data_input,
        batch_size=32,
        context_length=512,
    ):

        transformation = create_transformation(include_covariates=True)
        transformed_dataset = transformation(iter(test_data_input), is_train=False)
        instance_splitter = InstanceSplitter(
            target_field="target",
            is_pad_field="is_pad",
            start_field="start",
            forecast_start_field="forecast_start",
            instance_sampler=TestSplitSampler(),
            past_length=context_length,
            future_length=self.prediction_length,
            time_series_fields=[
                "observed_values",
                "feat_dynamic_real",
            ],
            dummy_value=np.nan,
        )
        iterables = [instance_splitter.apply(transformed_dataset, is_train=False)]
        iterators = list(map(iter, iterables))
        test_data_input_transformed = itertools.chain(*iterators)

        forecast_outputs = []
        for batch in tqdm(batcher(test_data_input_transformed, batch_size=batch_size)):
            context = [torch.tensor(entry["past_target"]) for entry in batch]
            covariates = [
                {
                    "future_feat_dynamic_real": entry.get("future_feat_dynamic_real", None),
                    "past_feat_dynamic_real": entry.get("past_feat_dynamic_real", None),
                }
                for entry in batch
            ]
            forecast_outputs.append(self.predict(context, covariates).numpy())

        forecast_outputs = np.concatenate(forecast_outputs)

        # Convert forecast samples into gluonts Forecast objects
        forecasts = []
        for item, ts in zip(forecast_outputs, test_data_input):
            forecast_start_date = ts["start"] + len(ts["target"])
            forecasts.append(SampleForecast(samples=item, start_date=forecast_start_date))

        return forecasts
