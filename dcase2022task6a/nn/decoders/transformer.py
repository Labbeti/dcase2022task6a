#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import math

from typing import Any, Optional, Union

import torch

from torch import nn, Tensor
from torch.nn import functional as F

from dcase2022task6a.nn.functional.misc import (
    generate_square_subsequent_mask,
    lengths_to_pad_mask,
    lengths_to_non_pad_mask,
    tensor_to_pad_mask,
    pad_sequence_rec,
    get_inverse_perm,
    sample_gumbels_like,
)
from dcase2022task6a.nn.modules.positional_encoding import PositionalEncoding
from dcase2022task6a.nn.modules.typical import TypicalLogitsWarper
from dcase2022task6a.utils.misc import list_dict_to_dict_list


class CustomTransformerDecoder(nn.Module):
    """
    Transformer Decoder module with teacher forcing, greedy search and generate (beam search) method.
    """

    def __init__(
        self,
        vocab_size: int,
        sos_idx: int,
        eos_idx: int,
        pad_idx: int,
        activation: str = "gelu",
        d_model: int = 256,
        dim_feedforward: int = 2048,
        dropout: float = 0.2,
        freeze_weight: str = "none",
        layer_norm_eps: float = 1e-5,
        max_output_size: int = 22,
        nhead: int = 4,
        num_decoder_layers: int = 6,
        use_memory_key_padding_mask: bool = True,
        share_proj_word_weights: bool = False,
        proj_layer_bias: bool = True,
        get_attns: bool = False,
        emb_scale_grad_by_freq: bool = False,
    ) -> None:
        if get_attns:
            raise NotImplementedError(f"Invalid argument {get_attns=}, this feature has been removed. (expected False)")

        super().__init__()
        self.vocab_size = vocab_size
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.max_output_size = max_output_size
        self.use_memory_key_padding_mask = use_memory_key_padding_mask
        self.get_attns = get_attns

        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.word_embed = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=pad_idx,
            scale_grad_by_freq=emb_scale_grad_by_freq,
        )

        tch_decoder_layer_cls = nn.TransformerDecoderLayer
        tch_decoder_cls = nn.TransformerDecoder

        decoder_layer = tch_decoder_layer_cls(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=False,
            norm_first=False,
        )
        self.tch_decoder = tch_decoder_cls(decoder_layer, num_decoder_layers)
        self.projection_layer = nn.Linear(d_model, vocab_size, bias=proj_layer_bias)

        if share_proj_word_weights:
            self.projection_layer.weight = self.word_embed.weight

        self.freeze(freeze_weight)

    def freeze(self, freeze_weight: str) -> None:
        if freeze_weight == "none":
            return None
        elif freeze_weight == "all":
            for _, param in self.named_parameters():
                param.requires_grad = False
        else:
            raise ValueError(
                f"Invalid argument mode {freeze_weight=} for {self.__class__.__name__}. Must be one of 'none' or 'all'"
            )

    def decode(
        self,
        frame_embs: Tensor,
        frame_embs_lens: Optional[Tensor],
        captions: Tensor,
        captions_pad_mask: Optional[Tensor] = None,
        return_dict: bool = False,
    ) -> Union[Tensor, dict[str, Any]]:
        """
        Scores next words with audio frame embs and previous words.

        :param frame_embs: (n_frames, bsize, embed_len)
        :param frame_embs_lens: (bsize,)
        :param captions: (caption_len, bsize)
        :param captions_pad_mask: (caption_len, bsize) or None
        :param return_dict: If True, return a dictionary containing the intermediade outputs of the model with "logits", "word_embs" and "attentions".
            Otherwise return only the logits.
        :returns: logits of shape (bsize, caption_len, vocab_size) or a dict of {"logits": ..., "word_embs": ..., "attentions": ...}
        """
        if (
            frame_embs_lens is not None
            and frame_embs.shape[1] != frame_embs_lens.shape[0]
        ):
            raise ValueError(
                f"Invalid bsize dim 1 for {frame_embs.shape=} for {self.__class__.__name__}.decode method."
            )

        captions_sq_mask = generate_square_subsequent_mask(
            captions.shape[0],
            captions.device,
        )

        captions = self.word_embed(captions) * math.sqrt(self.d_model)
        captions = self.pos_encoder(captions)

        if self.use_memory_key_padding_mask and frame_embs_lens is not None:
            # Note : include=False because we compute pad_mask and not a non_pad_mask
            frame_embs_pad_mask = lengths_to_pad_mask(
                frame_embs_lens,
                frame_embs.shape[0],
                include=False,
            )
        else:
            frame_embs_pad_mask = None

        tch_decoder_outs = self.tch_decoder(
            tgt=captions,
            tgt_mask=captions_sq_mask,
            tgt_key_padding_mask=captions_pad_mask,
            memory=frame_embs,
            memory_mask=None,
            memory_key_padding_mask=frame_embs_pad_mask,
        )

        if isinstance(tch_decoder_outs, Tensor):
            tch_decoder_outs = {"word_embs": tch_decoder_outs, "attentions": None}
        else:
            tch_decoder_outs["word_embs"] = tch_decoder_outs.pop("output")

        logits = self.projection_layer(tch_decoder_outs["word_embs"])

        if not return_dict:
            return logits
        else:
            return {
                "logits": logits,
            } | tch_decoder_outs

    def forward(
        self,
        frame_embs: Tensor,
        frame_embs_lens: Tensor,
        captions: Tensor,
        return_dict: bool = False,
    ) -> Union[Tensor, dict[str, Any]]:
        """
        Forward method for training the module.

        :param frame_embs: (bsize, embed_len, n_frames)
        :param frame_embs_lens: (bsize,)
        :param captions: (bsize, caption_len)
        :param return_dict: If True, return a dictionary containing the intermediade outputs of the model with "logits", "word_embs" and "attentions".
            Otherwise return only the logits.
        :returns: (max_output_size, bsize, vocab_size)
        """
        return self.teacher_forcing(frame_embs, frame_embs_lens, captions, return_dict)

    def teacher_forcing(
        self,
        frame_embs: Tensor,
        frame_embs_lens: Tensor,
        captions: Tensor,
        return_dict: bool = False,
    ) -> Union[Tensor, dict[str, Any]]:
        """
        :param frame_embs: (bsize, embed_len, n_frames)
        :param frame_embs_lens: (bsize,)
        :param captions: (bsize, caption_len)
        :param return_dict: If True, return a dictionary containing the intermediade outputs of the model with "logits", "word_embs" and "attentions".
            Otherwise return only the logits.
        :returns: (max_output_size, bsize, vocab_size)
        """
        self.__class__._check_teacher_forcing_args(**locals())

        # (bsize, embed_len, n_frames) -> (n_frames, bsize, embed_len)
        frame_embs = frame_embs.permute(2, 0, 1)

        captions_pad_mask = tensor_to_pad_mask(captions, pad_value=self.pad_idx)
        captions = captions.permute(1, 0)

        dec_outputs = self.decode(
            frame_embs,
            frame_embs_lens,
            captions,
            captions_pad_mask,
            return_dict=return_dict,
        )

        # permute logits : (max_output_size, bsize, vocab_size) -> (bsize, vocab_size, max_output_size)
        if return_dict:
            assert isinstance(dec_outputs, dict)
            dec_outputs["logits"] = dec_outputs["logits"].permute(1, 2, 0)
        else:
            assert isinstance(dec_outputs, Tensor)
            # dec_outputs is logits here
            dec_outputs = dec_outputs.permute(1, 2, 0)

        return dec_outputs

    def greedy_search(
        self,
        frame_embs: Tensor,
        frame_embs_lens: Tensor,
        return_dict: bool = False,
    ) -> Union[Tensor, dict[str, Any]]:
        """returns logits"""
        # (bsize, embed_size, n_frames) -> (n_frames, bsize, embed_size)
        frame_embs = frame_embs.permute(2, 0, 1)

        bsize = frame_embs_lens.shape[0]
        device = frame_embs.device
        dtype = frame_embs.dtype

        current_preds = torch.full(
            (1, bsize), self.sos_idx, device=device, dtype=torch.long
        )
        logits = torch.empty((0, bsize, self.vocab_size), dtype=dtype, device=device)
        attentions = []

        for _ in range(self.max_output_size - 1):
            # current_preds : (current_pred_size, bsize)
            dec_outputs = self.decode(
                frame_embs,
                frame_embs_lens,
                current_preds,
                return_dict=return_dict,
            )
            if isinstance(dec_outputs, Tensor):
                logits_i = dec_outputs
                attentions_i = None
            else:
                logits_i = dec_outputs["logits"]
                attentions_i = dec_outputs["attentions"]
            logits_i = logits_i[-1].unsqueeze(dim=0)
            preds_i = torch.argmax(logits_i, dim=-1)

            current_preds = torch.cat([current_preds, preds_i], dim=0)
            logits = torch.cat([logits, logits_i], dim=0)
            attentions.append(attentions_i)

        # logits : (max_output_size, bsize, vocab_size) -> (bsize, vocab_size, max_output_size)
        logits = logits.permute(1, 2, 0)
        if not return_dict:
            return logits
        else:
            return {
                "logits": logits,
                "preds": current_preds.permute(1, 0),
                "attentions": attentions,
            }

    def generate(
        self,
        frame_embs: Tensor,
        frame_embs_lens: Tensor,
        return_all_preds: bool = False,
        return_logits: bool = False,
        # Common search hparams
        use_gumbel: bool = False,
        temperature: Optional[float] = None,
        # Beam search hparams
        beam_size: int = 2,
        beam_alpha: float = 1.0,
        # Sampling hparams
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        typical_p: Optional[float] = None,
        generator: Union[None, int, torch.Generator] = None,
    ) -> dict[str, Any]:
        """
        :param frame_embs: (bsize, frame_emb_size, n_frames)
        :param frame_embs_lens: (bsize,)
        :param return_all_preds: If True, returns beam_size preds generated by the model.
        :param return_logits: If True, returns the logits of the best pred.
            Note : use argmax to re-build the best pred is incorrect because the word is not always the one with the maximal probability.

        :param use_gumbel: TODO
        :param temperature: TODO

        :param beam_size: The number of beams for the search.
        :param beam_alpha: TODO

        :param top_k: TODO
        :param top_p: TODO
        :param typical_p: TODO
        :param generator: The torch.Generator object used for sampling.
            If generator is an int, a new torch.Generator object will be created with this int as seed.
        :return: A dictionary containing :
            "preds": (bsize, max_seq_len)
            "scores": (bsize, max_seq_len)
            "best_beam_idx": int
        """
        if frame_embs.shape[0] == 1:
            return self.__class__._generate_single(**locals())
        else:
            outs: Any = []
            for i, (frame_embs_i, frame_embs_lens_i) in enumerate(
                zip(frame_embs, frame_embs_lens)
            ):
                frame_embs_i = frame_embs_i.unsqueeze(dim=0)
                frame_embs_lens_i = frame_embs_lens_i.unsqueeze(dim=0)
                out_i = self._generate_single(
                    frame_embs=frame_embs_i,
                    frame_embs_lens=frame_embs_lens_i,
                    return_all_preds=return_all_preds,
                    return_logits=return_logits,
                    # Logits processing hparams
                    use_gumbel=use_gumbel,
                    temperature=temperature,
                    # Beam search hparams
                    beam_size=beam_size,
                    beam_alpha=beam_alpha,
                    # Sampling hparams
                    top_k=top_k,
                    top_p=top_p,
                    typical_p=typical_p,
                    generator=generator,
                )
                outs.append(out_i)

            outs = list_dict_to_dict_list(outs, None)
            outs = {
                key: (
                    pad_sequence_rec(values, self.pad_idx)
                    if key.startswith("preds")
                    else values
                )
                for key, values in outs.items()
            }
            return outs

    def _check_teacher_forcing_args(
        self,
        frame_embs: Tensor,
        frame_embs_lens: Tensor,
        captions: Tensor,
        **kwargs,
    ) -> None:
        if (
            not isinstance(frame_embs, Tensor)
            or not isinstance(frame_embs_lens, Tensor)
            or not isinstance(captions, Tensor)
        ):
            raise TypeError(
                f"Invalid input types."
                f"Found ({frame_embs.__class__.__name__}, {frame_embs_lens.__class__.__name__}, {captions.__class__.__name__}) for (frame_embs, frame_embs_lens, captions) but expected (Tensor, Tensor, Tensor)"
            )
        if (
            frame_embs.ndim != 3
            or frame_embs_lens.ndim != 1
            or captions.ndim != 2
            or frame_embs.shape[0] != frame_embs_lens.shape[0]
            or frame_embs.shape[0] != captions.shape[0]
        ):
            raise ValueError(
                f"Invalid input shapes for {self.__class__.__name__}. (found {frame_embs.ndim=}, {frame_embs_lens.ndim=}, {captions.ndim=}, expected (3, 1, 2))"
            )

    def _check_generate_single_args(
        self,
        frame_embs: Tensor,
        frame_embs_lens: Tensor,
        beam_size: int,
        top_k: Optional[int],
        top_p: Optional[float],
        typical_p: Optional[float],
        **kwargs,
    ) -> None:
        n_sample_args = sum(arg is not None for arg in (top_k, top_p, typical_p))
        if n_sample_args not in (0, 1):
            raise ValueError(
                f"Invalid arguments ({top_k=}, {top_p=}, {typical_p=}). (found {n_sample_args} args set but expected 0 or 1)"
            )
        if frame_embs.shape[0] != 1:
            raise ValueError(
                f"Argument shape {frame_embs.shape=} != 1 is not supported by {self.__class__.__name__}.generate method."
            )
        if frame_embs_lens.shape[0] != 1:
            raise ValueError(
                f"Argument shape {frame_embs_lens.shape=} != 1 is not supported by {self.__class__.__name__}.generate method."
            )
        if beam_size < 1 or self.vocab_size < beam_size:
            raise ValueError(
                f"Invalid argument beam_size. (found {beam_size=} < 1 or vocab_size={self.vocab_size} < {beam_size=})"
            )
        if top_k is not None and top_k < 1:
            raise ValueError(
                f"Invalid argument {top_k=}, it must be strictly positive or None."
            )
        if top_p is not None and (top_p < 0 or top_p > 1.0):
            raise ValueError(
                f"Invalid argument {top_p=}, it must be in range [0, 1] or None."
            )
        if typical_p is not None and (typical_p < 0 or typical_p > 1.0):
            raise ValueError(
                f"Invalid argument {typical_p=}, it must be in range [0, 1] or None."
            )

    def _generate_single(
        self,
        frame_embs: Tensor,
        frame_embs_lens: Tensor,
        return_all_preds: bool = False,
        return_logits: bool = False,
        # Logits processing hparams
        use_gumbel: bool = False,
        temperature: Optional[float] = None,
        # Beam search hparams
        beam_size: int = 2,
        beam_alpha: float = 1.0,
        # Sampling hparams
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        typical_p: Optional[float] = None,
        generator: Union[None, int, torch.Generator] = None,
    ) -> dict[str, Any]:
        self.__class__._check_generate_single_args(**locals())
        n_sample_args = sum(arg is not None for arg in (top_k, top_p, typical_p))
        do_sample = n_sample_args >= 1
        device = frame_embs.device

        if isinstance(generator, int):
            generator = torch.Generator(device=device).manual_seed(generator)

        # (bsize, embed_len, n_frames) -> (n_frames, bsize, embed_size)
        frame_embs = frame_embs.permute(2, 0, 1)

        n_frames, _, embed_size = frame_embs.shape

        current_beam_size = beam_size
        sum_logprobs = None
        # current_preds : shape is (cap_len, current_beam)
        current_preds = torch.full(
            (1, current_beam_size), self.sos_idx, dtype=torch.long, device=device
        )
        current_logits = torch.empty(
            (0, current_beam_size, self.vocab_size), dtype=torch.float, device=device
        )

        finished_mean_logprobs_lst = []
        finished_preds_lst = []
        finished_logits_lst = []

        frame_embs_expanded = frame_embs.expand(
            n_frames, current_beam_size, embed_size
        ).contiguous()
        frame_embs_lens_expanded = frame_embs_lens.expand(
            current_beam_size
        ).contiguous()

        for i in range(self.max_output_size):
            frame_embs_expanded = frame_embs_expanded[:, :current_beam_size]
            frame_embs_lens_expanded = frame_embs_lens_expanded[:current_beam_size]

            all_logits_i = self.decode(
                frame_embs_expanded,
                frame_embs_lens_expanded,
                current_preds,
                return_dict=False,
            )
            assert isinstance(all_logits_i, Tensor)

            # all_logits_i : shape is (cap_len, current_beam, vocab_size)
            logits_i = all_logits_i[-1]

            if use_gumbel:
                logits_i = logits_i + sample_gumbels_like(logits_i, generator)

            if temperature is not None:
                logits_i = logits_i / temperature

            if not do_sample:
                prev_beam_idxs, next_word_idxs, sum_logprobs = self._select_next_tokens(
                    logits_i,
                    sum_logprobs,
                )
            else:
                prev_beam_idxs, next_word_idxs, sum_logprobs = self._sample_next_tokens(
                    logits_i,
                    sum_logprobs,
                    top_k,
                    top_p,
                    typical_p,
                    generator,
                )

            # Update current values
            current_preds = torch.cat(
                (current_preds[:, prev_beam_idxs], next_word_idxs.unsqueeze(dim=0)),
                dim=0,
            )
            current_logits = torch.cat(
                (
                    current_logits[:, prev_beam_idxs],
                    logits_i[prev_beam_idxs].unsqueeze(dim=0),
                ),
                dim=0,
            )

            if i < self.max_output_size - 1:
                mask_finished_beam = next_word_idxs == self.eos_idx
            else:
                mask_finished_beam = torch.ones(
                    next_word_idxs.shape[0],
                    dtype=torch.bool,
                    device=next_word_idxs.device,
                )
            mask_unfinished_beam = mask_finished_beam.logical_not()

            if mask_finished_beam.sum() > 0:
                # Score of a prediction is sum(log(probs)) / len(prediction)
                finished_mean_logprobs_lst += (
                    sum_logprobs[mask_finished_beam]
                    .div((current_preds.shape[0] - 1) ** beam_alpha)
                    .tolist()
                )
                finished_preds_lst += list(
                    current_preds[:, mask_finished_beam].permute(1, 0)
                )
                finished_logits_lst += list(
                    current_logits[:, mask_finished_beam].permute(1, 0, 2)
                )

            sum_logprobs = sum_logprobs[mask_unfinished_beam]
            current_preds = current_preds[:, mask_unfinished_beam]
            current_logits = current_logits[:, mask_unfinished_beam]
            current_beam_size = int(mask_unfinished_beam.sum().item())

            if current_beam_size == 0:
                break

        finished_mean_logprobs_lst = torch.as_tensor(
            finished_mean_logprobs_lst, device=device
        )
        best_beam_idx = finished_mean_logprobs_lst.argmax()
        best_mean_logprobs = finished_mean_logprobs_lst[best_beam_idx]

        # Note [1:] : remove <sos> in the beginning of best_pred
        best_pred = torch.as_tensor(
            finished_preds_lst[best_beam_idx][1:],
            device=device,
        )
        best_logits = finished_logits_lst[best_beam_idx]
        log_activation = nn.LogSoftmax(dim=-1)
        best_logprobs = log_activation(best_logits)
        best_scores = best_logprobs.gather(1, best_pred.unsqueeze(dim=1)).squeeze(dim=1)

        # Sanity check for default score
        best_score_from_logprobs = best_scores.mean()
        if (
            not do_sample
            and (beam_alpha is None or beam_alpha == 1.0)
            and not torch.allclose(best_score_from_logprobs, best_mean_logprobs)
        ):
            logging.error(
                f"INTERNAL ERROR: Invalid best score in beam search. ({best_score_from_logprobs=} != {best_mean_logprobs=}, {best_beam_idx.item()=})"
            )

        # The key 'preds' is important for Evaluator callback, it defines the main output of the model
        outs = {
            "preds": best_pred,
            "scores": best_scores,
            "best_beam_idx": best_beam_idx.item(),
        }
        if return_all_preds:
            outs |= {
                "preds_all_beams": finished_preds_lst,
                "score_all_beams": finished_mean_logprobs_lst,
            }
        if return_logits:
            outs["logits"] = best_logits
        if return_logits and return_all_preds:
            outs["logits_all_beams"] = finished_logits_lst
        return outs

    def _select_next_tokens(
        self,
        logits_i: Tensor,
        prev_sum_logprobs: Optional[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        :param logits_i: (current_beam_size, vocab_size)
        :param prev_sum_logprobs: (current_beam_size,)
        """
        current_beam_size, vocab_size = logits_i.shape
        log_activation = nn.LogSoftmax(dim=-1)
        logprobs_i = log_activation(logits_i)

        if prev_sum_logprobs is None:
            sum_logprobs = logprobs_i[0].unsqueeze(dim=0)
        else:
            prev_sum_logprobs = prev_sum_logprobs.unsqueeze(dim=1).expand(
                current_beam_size,
                vocab_size,
            )
            sum_logprobs = prev_sum_logprobs + logprobs_i

        sum_logprobs_flat = sum_logprobs.view(-1)
        sum_logprobs_selected, next_token_idxs_flat = torch.topk(
            sum_logprobs_flat, current_beam_size
        )

        prev_beam_idxs = next_token_idxs_flat.div(
            vocab_size,
            rounding_mode="trunc",
        )
        next_word_idxs = next_token_idxs_flat % vocab_size

        # prev_beam_idxs: shape is (current_beam,), values in [0, current_beam[
        # next_word_idxs: shape is (current_beam,), values in [0, vocab_size[
        # sum_logprobs_selected: shape is (current_beam,), values in ]-inf, 0]

        return prev_beam_idxs, next_word_idxs, sum_logprobs_selected

    def _sample_next_tokens(
        self,
        logits_i: Tensor,
        prev_sum_logprobs: Optional[Tensor],
        # Sampling args
        top_k: Optional[int],
        top_p: Optional[float],
        typical_p: Optional[float],
        generator: Union[None, torch.Generator],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        :param logits_i: (current_beam_size, vocab_size)
        :param prev_sum_logprobs: (current_beam_size,)
        """
        current_beam_size, vocab_size = logits_i.shape
        log_activation = nn.LogSoftmax(dim=-1)

        if prev_sum_logprobs is None:
            logits_i = logits_i[0].unsqueeze(dim=0)

        logprobs_i = log_activation(logits_i)
        probs_i = logprobs_i.exp()

        if prev_sum_logprobs is None:
            sum_logprobs = logprobs_i
        else:
            prev_sum_logprobs = prev_sum_logprobs.unsqueeze(dim=1).expand(
                current_beam_size,
                vocab_size,
            )
            sum_logprobs = prev_sum_logprobs + logprobs_i

        # probs: (current_beam_size, vocab_size)
        # sum_logprobs: (current_beam_size, vocab_size)

        if top_k is not None:
            probs_sorted_masked, indexes = probs_i.sort(dim=-1, descending=True)
            # Set other probabilities below top_k to 0
            probs_sorted_masked[:, top_k:] = min(probs_sorted_masked.min().item(), 0.0)

            indexes_inv = get_inverse_perm(indexes)
            probs_masked = probs_sorted_masked.gather(1, indexes_inv)
            probs_masked = F.normalize(probs_masked, dim=-1, p=1)

        elif top_p is not None:
            probs_sorted_masked, indexes = probs_i.sort(dim=-1, descending=True)
            probs_cumsum = probs_sorted_masked.cumsum(dim=-1)
            top_p_ends = (probs_cumsum >= top_p).int().argmax(dim=-1) + 1
            # top_p_ends : (current_beam,)
            mask = lengths_to_non_pad_mask(
                top_p_ends,
                max_len=probs_i.shape[1],
                include=True,
            )
            probs_sorted_masked = probs_sorted_masked * mask

            indexes_inv = get_inverse_perm(indexes)
            probs_masked = probs_sorted_masked.gather(1, indexes_inv)
            probs_masked = F.normalize(probs_masked, dim=-1, p=1)

        elif typical_p is not None:
            warper = TypicalLogitsWarper(
                mass=typical_p,
                min_tokens_to_keep=current_beam_size,
            )
            logits_i = warper(None, logits_i)
            logprobs_i = log_activation(logits_i)
            probs_masked = logprobs_i.exp()

        else:
            raise ValueError(f"Invalid arguments ({top_k=}, {top_p=}, {typical_p=})")

        # probs_masked : (current_beam, vocab_size)

        sampled_indexes = torch.multinomial(
            input=probs_masked,
            num_samples=current_beam_size,
            replacement=False,
            generator=generator,
        )
        # sampled_indexes: (current_beam, vocab_size)

        sum_logprobs_masked = torch.full_like(sum_logprobs, -math.inf)
        sum_logprobs_masked.scatter_(
            1, sampled_indexes, sum_logprobs.gather(1, sampled_indexes)
        )

        sum_logprobs_flat = sum_logprobs_masked.view(-1)
        sum_logprobs_selected, next_token_idxs_flat = torch.topk(
            sum_logprobs_flat, current_beam_size
        )

        prev_beam_idxs = next_token_idxs_flat.div(
            vocab_size,
            rounding_mode="trunc",
        )
        next_word_idxs = next_token_idxs_flat % vocab_size

        return prev_beam_idxs, next_word_idxs, sum_logprobs_selected
