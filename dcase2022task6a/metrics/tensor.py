#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Sequence

import torch

from torch import nn, Tensor

from dcase2022task6a.nn.functional.misc import tensor_to_lengths


class MeanPredLen(nn.Module):
    def __init__(self, eos_idx: int) -> None:
        super().__init__()
        self.eos_idx = eos_idx

    def forward(self, preds: Tensor) -> float:
        assert preds.ndim == 2
        lens = tensor_to_lengths(preds, end_value=self.eos_idx)
        return lens.float().mean().item()


class TensorDiversity1(nn.Module):
    def __init__(self, exclude: Sequence[int] = ()) -> None:
        super().__init__()
        self.exclude = exclude

    def forward(self, preds: Tensor) -> float:
        assert preds.ndim == 2
        score = 0
        for pred in preds:
            tokens_set = pred.unique().tolist()
            tokens_set = [token for token in tokens_set if token not in self.exclude]
            score += len(tokens_set) / len(pred)
        score /= len(preds)
        return score


class GlobalTensorVocabUsage(nn.Module):
    r"""Global Vocab Usage.

    Returns \frac{|hyp\_vocab|}{|ref\_vocab|}
    """

    def __init__(self, ignored_indexes: Sequence[int]) -> None:
        super().__init__()
        self._ignored_indexes = ignored_indexes
        self._preds_vocab = None
        self._captions_vocab = None

    def reset(self) -> None:
        self._preds_vocab = None
        self._captions_vocab = None

    def forward(self, preds: Tensor, captions: Tensor) -> float:
        """
        :param preds: (bsize, pred_len) tensor
        :param captions: (bsize, capt_len) tensor
        """
        self.update(preds, captions)
        return self.compute()

    def update(self, preds: Tensor, captions: Tensor) -> None:
        preds = preds[preds == self._ignored_indexes]
        captions = captions[captions == self._ignored_indexes]

        preds_vocab = torch.unique(preds)
        captions_vocab = torch.unique(captions)

        if self._preds_vocab is None:
            self._preds_vocab = preds_vocab
        else:
            self._preds_vocab = torch.unique(torch.cat(self._preds_vocab, preds_vocab))

        if self._captions_vocab is None:
            self._captions_vocab = captions_vocab
        else:
            self._captions_vocab = torch.unique(
                torch.cat(self._captions_vocab, captions_vocab)
            )

    def compute(self) -> float:
        if (
            self._preds_vocab is not None
            and self._captions_vocab is not None
            and len(self._captions_vocab) > 0
        ):
            return len(self._preds_vocab) / len(self._captions_vocab)
        else:
            return 0.0
