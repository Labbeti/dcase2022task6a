#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from typing import Any, Iterable, Optional, Sized, Union

import torch

from pytorch_lightning import LightningModule
from torch import nn, Tensor

from dcase2022task6a.metrics.cider_d import CiderD
from dcase2022task6a.metrics.tensor import MeanPredLen, TensorDiversity1
from dcase2022task6a.nn.decoders.transformer import CustomTransformerDecoder
from dcase2022task6a.nn.encoders.cnn10 import CNN10
from dcase2022task6a.tokenization import (
    AACTokenizer,
    EOS_TOKEN,
    SOS_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
)
from dcase2022task6a.utils.optimizers import get_optimizer
from dcase2022task6a.utils.schedulers import get_scheduler


logger = logging.getLogger(__name__)


class LightningCNN10Transformer(LightningModule):
    """CNNTRF (Convolutional Neural Network encoder and TRansFormer decoder) LightningModule."""

    def __init__(
        self,
        # Model params
        label_smoothing: float = 0.1,
        monitors: Iterable[str] = ("cider",),
        # Encoder params
        pretrained_encoder: bool = True,
        lens_rounding_mode: str = "trunc",
        waveform_input: bool = True,
        window_size: int = 1024,
        hop_size: int = 320,
        freeze_encoder: str = "none",
        # Decoder params
        max_output_size: Optional[int] = None,
        nhead: int = 4,
        d_model: int = 256,
        num_decoder_layers: int = 6,
        decoder_dropout_p: float = 0.2,
        dim_feedforward: int = 2048,
        activation: str = "gelu",
        # Decoding params
        return_all_preds: bool = False,
        use_gumbel: bool = False,
        temperature: Optional[float] = None,
        beam_size: int = 10,
        beam_alpha: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        typical_p: Optional[float] = None,
        generator: Union[int, None] = None,
        # Optimizer params
        optim_name: str = "Adam",
        lr: float = 5e-4,
        weight_decay: float = 1e-6,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        # Scheduler params
        sched_name: str = "cos_decay",
        sched_n_steps: Optional[int] = None,
        # Other params
        fit_tokenizer: Optional[AACTokenizer] = None,
        verbose: int = 0,
    ) -> None:
        super().__init__()
        # Model params
        self.label_smoothing = label_smoothing
        self.monitors = monitors
        # Encoder params
        self.pretrained_encoder = pretrained_encoder
        self.lens_rounding_mode = lens_rounding_mode
        self.waveform_input = waveform_input
        self.window_size = window_size
        self.hop_size = hop_size
        self.freeze_encoder = freeze_encoder
        # Decoder params
        self.max_output_size = max_output_size
        self.nhead = nhead
        self.d_model = d_model
        self.num_decoder_layers = num_decoder_layers
        self.decoder_dropout_p = decoder_dropout_p
        self.dim_feedforward = dim_feedforward
        self.activation = activation
        # Decoding params
        self.return_all_preds = return_all_preds
        self.use_gumbel = use_gumbel
        self.temperature = temperature
        self.beam_size = beam_size
        self.beam_alpha = beam_alpha
        self.top_k = top_k
        self.top_p = top_p
        self.typical_p = typical_p
        self.generator = generator
        # Optimizer params
        self.optim_name = optim_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps
        # Scheduler params
        self.sched_name = sched_name
        self.sched_n_steps = sched_n_steps
        # Other params
        self.fit_tokenizer = fit_tokenizer
        self.verbose = verbose

        self.criterion: nn.Module = nn.Identity()
        self.encoder: nn.Module = nn.Identity()
        self.decoder: CustomTransformerDecoder = None  # type: ignore
        self.projection: nn.Module = nn.Identity()
        self.fit_tensor_metrics = nn.ModuleDict()

        val_metrics = {}
        for monitor in monitors:
            if monitor == "cider":
                metric = CiderD()
            else:
                MONITOR_NAMES = ("cider",)
                raise ValueError(
                    f"Invalid monitor name {monitor=}. (expected one of {MONITOR_NAMES})"
                )
            val_metrics[monitor] = metric
        self.val_metrics = nn.ModuleDict(val_metrics)

        self._model_is_setup = False
        self._test_is_setup = False

        self.save_hyperparameters(ignore=("fit_tokenizer",))

        if self.fit_tokenizer is not None and self.fit_tokenizer.is_fit():
            self.setup()

    def load_state_dict(self, state_dict: Any, strict: bool = True) -> Any:
        def replace_key(key: str) -> str:
            return key.replace("model.encoder", "encoder").replace(
                "model.decoder", "decoder"
            )

        state_dict = {replace_key(key): value for key, value in state_dict.items()}
        return super().load_state_dict(state_dict, strict)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            if self.trainer is not None and self.trainer.datamodule is not None:  # type: ignore
                dataloader = self.trainer.datamodule.train_dataloader()  # type: ignore
                batch = next(iter(dataloader))
                if self.verbose >= 1:
                    logger.debug(f"Batch keys: {tuple(batch.keys())}")

                def get_shape(v) -> tuple[int, ...]:
                    if isinstance(v, Tensor):
                        return tuple(v.shape)
                    elif isinstance(v, Sized):
                        return (len(v),)
                    else:
                        return ()

                if self.verbose >= 1:
                    shapes = tuple(get_shape(v) for v in batch.values())
                    logger.debug(f"Batch shapes: {shapes}")
                audio, audio_shape, captions = (
                    batch["audio"],
                    batch["audio_shape"],
                    batch["captions"],
                )
                self.example_input_array = dict(
                    audio=audio,
                    audio_shape=audio_shape,
                    captions=captions,
                    decode_method="forcing",
                )

        if stage in ("test", None):
            self._test_is_setup = True

        if not self._model_is_setup:
            self._model_is_setup = True

            fit_tokenizer = self.fit_tokenizer
            if (
                not isinstance(fit_tokenizer, AACTokenizer)
                or not fit_tokenizer.is_fit()
            ):
                raise RuntimeError("AACTokenizer is not fit in CNNTRF.")

            self.fit_sos_idx = fit_tokenizer.stoi(SOS_TOKEN)
            self.fit_eos_idx = fit_tokenizer.stoi(EOS_TOKEN)
            self.fit_pad_idx = fit_tokenizer.stoi(PAD_TOKEN)
            self.fit_unk_idx = fit_tokenizer.stoi(UNK_TOKEN)

            excluded_indexes = [
                self.fit_sos_idx,
                self.fit_eos_idx,
                self.fit_pad_idx,
                self.fit_unk_idx,
            ]
            self.fit_tensor_metrics = nn.ModuleDict(
                {
                    "pred_len": MeanPredLen(self.fit_eos_idx),
                    "pred_diversity1": TensorDiversity1(excluded_indexes),
                }
            )

            if self.max_output_size is None:
                self.max_output_size = fit_tokenizer.get_max_sentence_size()
                if self.verbose >= 1:
                    logger.debug(f"Auto-detect value {self.max_output_size=}.")

            self.criterion = nn.CrossEntropyLoss(
                ignore_index=self.fit_pad_idx,
                label_smoothing=self.label_smoothing,
            )

            self.encoder = CNN10(
                add_clip_linear=False,
                add_frame_linear=True,
                fmax=14000,
                fmin=50,
                frame_emb_size=self.d_model,
                freeze_weight=self.freeze_encoder,
                hop_size=self.hop_size,
                lens_rounding_mode=self.lens_rounding_mode,
                mel_bins=64,
                pretrained=self.pretrained_encoder,
                sr=32000,
                use_spec_augment=False,
                waveform_input=self.waveform_input,
                window_size=self.window_size,
            )

            self.decoder = CustomTransformerDecoder(
                vocab_size=fit_tokenizer.get_vocab_size(),
                sos_idx=self.fit_sos_idx,
                eos_idx=self.fit_eos_idx,
                pad_idx=self.fit_pad_idx,
                max_output_size=self.max_output_size,
                nhead=self.nhead,
                d_model=self.d_model,
                num_decoder_layers=self.num_decoder_layers,
                dropout=self.decoder_dropout_p,
                dim_feedforward=self.dim_feedforward,
                activation=self.activation,
                use_memory_key_padding_mask=True,
                share_proj_word_weights=False,
                freeze_weight="none",
                get_attns=False,
                emb_scale_grad_by_freq=False,
            )

    def encode_and_proj(self, audio: Tensor, audio_shape: Tensor) -> dict[str, Tensor]:
        encoder_outs = self.encoder(audio, audio_shape)
        encoder_outs["frame_embs"] = self.projection(encoder_outs["frame_embs"])
        return encoder_outs

    def decode_forcing(
        self, frame_embs: Tensor, frame_embs_lens: Tensor, captions: Tensor
    ) -> Tensor:
        logits = self.decoder(frame_embs, frame_embs_lens, captions=captions)
        return logits

    def decode_greedy(self, frame_embs: Tensor, frame_embs_lens: Tensor) -> Tensor:
        logits = self.decoder.greedy_search(
            frame_embs, frame_embs_lens, return_dict=False
        )
        return logits  # type: ignore

    def decode_generate(
        self,
        frame_embs: Tensor,
        frame_embs_lens: Tensor,
        **kwargs,
    ) -> dict[str, Any]:
        gen_hparams = {
            "return_all_preds": self.return_all_preds,
            "use_gumbel": self.use_gumbel,
            "temperature": self.temperature,
            "beam_size": self.beam_size,
            "beam_alpha": self.beam_alpha,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "typical_p": self.typical_p,
            "generator": self.generator,
        }
        kwargs = gen_hparams | kwargs
        outs = self.decoder.generate(frame_embs, frame_embs_lens, **kwargs)
        return outs

    def training_step(self, batch: dict[str, Any], *args, **kwargs) -> Tensor:
        audio, audio_shape, captions = [
            batch[name] for name in ("audio", "audio_shape", "captions")
        ]

        # logits : (bsize, vocab_size, max_cap_size)
        encoder_outs = self.encode_and_proj(audio, audio_shape)
        logits = self.decode_forcing(**encoder_outs, captions=captions)

        # Remove SOS from captions for criterion
        captions = captions[:, 1:]
        logits = logits[:, :, : captions.shape[1]]
        loss = self.criterion(logits, captions)

        with torch.no_grad():
            prefix = "train"
            self.log(
                f"{prefix}/loss",
                loss,
                on_epoch=True,
                on_step=False,
                batch_size=audio.shape[0],
            )

            preds = logits.argmax(dim=1)
            scores = {
                f"{prefix}/{key}": metric(preds)
                for key, metric in self.fit_tensor_metrics.items()
            }
            self.log_dict(
                scores, on_epoch=True, on_step=False, batch_size=audio.shape[0]
            )

        return loss

    def validation_step(self, batch: dict[str, Any], *args, **kwargs) -> Any:
        audio, audio_shape, mult_captions = [
            batch[name] for name in ("audio", "audio_shape", "captions")
        ]

        losses = torch.zeros(
            size=(mult_captions.shape[1],),
            dtype=audio.dtype,
            device=audio.device,
        )

        scores_lsts = {key: [] for key in self.fit_tensor_metrics.keys()}
        encoder_outs = self.encode_and_proj(audio, audio_shape)

        for i in range(mult_captions.shape[1]):
            captions_i = mult_captions[:, i, :]

            # logits : (bsize, vocab_size, capt_len)
            logits = self.decode_forcing(**encoder_outs, captions=captions_i)
            # Remove SOS from captions for criterion
            captions_i = captions_i[:, 1:]
            logits = logits[:, :, : captions_i.shape[1]]
            losses[i] = self.criterion(logits, captions_i)

            preds = logits.argmax(dim=1)
            scores_lsts = {
                key: scores_lsts[key] + [metric(preds)]
                for key, metric in self.fit_tensor_metrics.items()
            }

        prefix = "val"
        loss = losses.mean()
        scores = {
            f"{prefix}/{key}": torch.as_tensor(lst, dtype=torch.float64).mean()
            for key, lst in scores_lsts.items()
        }

        self.log(
            f"{prefix}/loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            batch_size=audio.shape[0],
        )
        self.log_dict(scores, on_epoch=True, on_step=False, batch_size=audio.shape[0])

        if len(self.val_metrics) > 0:
            assert self.fit_tokenizer is not None
            preds = self.decode_generate(**encoder_outs)["preds"]
            hypotheses = self.fit_tokenizer.decode_batch(preds)
            mult_references = self.fit_tokenizer.decode_rec(mult_captions)
            return {"hypotheses": hypotheses, "references": mult_references}
        else:
            return {}

    def validation_epoch_end(self, outs: list[dict]) -> None:
        if len(self.val_metrics) > 0:
            all_hypotheses = [hyp for out in outs for hyp in out["hypotheses"]]
            all_references = [refs for out in outs for refs in out["references"]]

            prefix = "val"
            scores = {
                f"{prefix}/{metric_name}": metric(all_hypotheses, all_references)
                for metric_name, metric in self.val_metrics.items()
            }
            self.log_dict(scores, on_step=False, on_epoch=True)

    def test_step(self, batch: dict[str, Any], *args, **kwargs) -> dict[str, Any]:
        audio, audio_shape, captions = [
            batch[name] for name in ("audio", "audio_shape", "captions")
        ]
        gen_outs = self(audio, audio_shape, decode_method="generate")

        assert self.fit_tokenizer is not None
        for outs in (gen_outs,):
            for name in outs.keys():
                if name.startswith("preds"):
                    outs[name] = self.fit_tokenizer.decode_rec(
                        outs[name],
                    )

        return gen_outs | {"captions": captions}

    def forward(
        self,
        audio: Tensor,
        audio_shape: Tensor,
        decode_method: str = "generate",
        **kwargs,
    ) -> Union[Tensor, dict]:
        encoder_outs = self.encode_and_proj(audio, audio_shape)

        if decode_method == "forcing":
            if "captions" not in kwargs.keys():
                raise ValueError(
                    f"Please provides 'captions' keyword argument with {decode_method=}."
                )
            logits = self.decode_forcing(**encoder_outs, **kwargs)
            outs = logits

        elif decode_method == "greedy":
            logits = self.decode_greedy(**encoder_outs, **kwargs)
            outs = logits

        elif decode_method == "generate":
            outs = self.decode_generate(**encoder_outs, **kwargs)
        else:
            raise ValueError(
                f"Unknown argument {decode_method=}. (expected 'forcing', 'greedy' or 'generate')"
            )

        return outs

    def configure_optimizers(self) -> tuple[list, list]:
        optimizer = get_optimizer(
            self.optim_name,
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
            eps=self.eps,
        )
        if self.sched_n_steps is not None:
            sched_n_steps = self.sched_n_steps
        elif self.trainer is not None and isinstance(self.trainer.max_epochs, int):
            sched_n_steps = self.trainer.max_epochs
        else:
            raise RuntimeError(
                f"Cannot get param 'sched_n_steps' from Trainer. ({self.sched_n_steps=}, {self.trainer=})"
            )
        scheduler = get_scheduler(
            self.sched_name,
            optimizer,
            sched_n_steps=sched_n_steps,
        )
        return [optimizer], ([scheduler] if scheduler is not None else [])
