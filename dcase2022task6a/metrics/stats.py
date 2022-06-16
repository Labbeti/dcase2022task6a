#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter
from typing import Union

import torch

from torch import nn


class GlobalVocabUsage(nn.Module):
    r"""Global Vocab Usage.

    Returns \frac{|hyp\_vocab|}{|ref\_vocab|}
    """

    def forward(
        self, hypotheses: list[list[str]], references: list[list[list[str]]]
    ) -> float:
        hypotheses_vocab = dict.fromkeys(token for hyp in hypotheses for token in hyp)
        references_vocab = dict.fromkeys(
            token for refs in references for ref in refs for token in ref
        )
        if len(references_vocab) > 0:
            return len(hypotheses_vocab) / len(references_vocab)
        else:
            return 0.0


class GlobalVocabCoverage(nn.Module):
    r"""Global Vocabulary Coverage.
    Every word in hypothesis vocabulary is weighted by its frequency in references vocabulary.

    Returns \frac{sum_{w \in hyp\_vocab} occurrence\_in\_refs[w]}{nb\_words\_in\_refs}
    """

    def forward(
        self, hypotheses: list[list[str]], references: list[list[list[str]]]
    ) -> float:
        hyp_freqs = Counter()
        hyp_freqs.update(word for hyp in hypotheses for word in hyp)

        ref_vocab = Counter()
        references_flatten = [
            word for refs in references for ref in refs for word in ref
        ]
        ref_vocab.update(references_flatten)

        if len(references_flatten) > 0:
            return sum(ref_vocab[word] for word in hyp_freqs.keys()) / len(
                references_flatten
            )
        else:
            return 0.0


class GlobalVocabFreq(nn.Module):
    def forward(
        self, hypotheses: list[list[str]], references: list[list[list[str]]]
    ) -> float:
        hyp_freqs = Counter()
        hyp_freqs.update(word for hyp in hypotheses for word in hyp)

        ref_vocab = Counter()
        references_flatten = [
            word for refs in references for ref in refs for word in ref
        ]
        ref_vocab.update(references_flatten)

        total = sum(ref_vocab.values())
        if len(hyp_freqs) > 0 and total > 0:
            return sum(ref_vocab[word] / total for word in hyp_freqs.keys()) / len(
                hyp_freqs
            )
        else:
            return 0.0


class HypMeanLen(nn.Module):
    """Global"""

    def forward(
        self,
        hypotheses: list[list[str]],
        references: list[list[list[str]]],
        return_dict: bool = False,
    ) -> Union[float, dict]:
        dtype = torch.float64
        lens = [len(hyp) for hyp in hypotheses]
        lens = torch.as_tensor(lens, dtype=dtype)
        mean_len = lens.mean().item()

        if not return_dict:
            return mean_len
        else:
            return {
                "score": mean_len,
                "scores": lens,
            }


class RefMeanLen(nn.Module):
    """Global"""

    def forward(
        self, hypotheses: list[list[str]], references: list[list[list[str]]]
    ) -> float:
        ref_lens = [len(ref) for refs in references for ref in refs]
        dtype = torch.float64
        ref_mean = torch.as_tensor(ref_lens, dtype=dtype).mean().item()
        return float(ref_mean)


class HypVocabLen(nn.Module):
    """Global"""

    def forward(
        self, hypotheses: list[list[str]], references: list[list[list[str]]]
    ) -> float:
        vocab = set(token for hyp in hypotheses for token in hyp)
        return float(len(vocab))


class RefVocabLen(nn.Module):
    """Global"""

    def forward(
        self, hypotheses: list[list[str]], references: list[list[list[str]]]
    ) -> float:
        vocab = set(token for refs in references for ref in refs for token in ref)
        return float(len(vocab))


class GlobalVocabPrecision(nn.Module):
    """Global.
    Score in [0, 1]
    """

    def forward(
        self, hypotheses: list[list[str]], references: list[list[list[str]]]
    ) -> float:
        hypotheses_vocab = dict.fromkeys(token for hyp in hypotheses for token in hyp)
        references_vocab = dict.fromkeys(
            token for refs in references for ref in refs for token in ref
        )
        global_hyp_in_refs = len(
            [
                token
                for token in hypotheses_vocab.keys()
                if token in references_vocab.keys()
            ]
        )
        score = (
            global_hyp_in_refs / len(hypotheses_vocab)
            if len(hypotheses_vocab) > 0
            else 0
        )
        return score


class LocalVocabPrecision(nn.Module):
    """Local.
    Score in [0, 1]
    """

    def forward(
        self, hypotheses: list[list[str]], references: list[list[list[str]]]
    ) -> float:
        n_hyp_in_refs = []

        for hyp, refs in zip(hypotheses, references):
            hyp_vocab = dict.fromkeys(hyp)
            refs_vocab = dict.fromkeys(token for ref in refs for token in ref)

            local_hyp_in_refs = len(
                [token for token in hyp_vocab.keys() if token in refs_vocab.keys()]
            )
            score = local_hyp_in_refs / len(hyp_vocab) if len(hyp_vocab) > 0 else 0
            n_hyp_in_refs.append(score)

        score = sum(n_hyp_in_refs) / len(n_hyp_in_refs)
        return score


class Len(nn.Module):
    """Global"""

    def forward(self, hypotheses: list, references: list) -> float:
        return float(len(references))


class HypVocabUsage(nn.Module):
    """Global"""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self._vocab_size = vocab_size
        self._hyp_vocab_len = HypVocabLen()

    def forward(
        self, hypotheses: list[list[str]], references: list[list[list[str]]]
    ) -> float:
        hyp_vocab_len = self._hyp_vocab_len(hypotheses, None)
        return hyp_vocab_len / self._vocab_size


class RefVocabUsage(nn.Module):
    """Global"""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self._vocab_size = vocab_size
        self._hyp_vocab_len = HypVocabLen()
        self._ref_vocab_len = RefVocabLen()

    def forward(
        self, hypotheses: list[list[str]], references: list[list[list[str]]]
    ) -> float:
        ref_vocab_len = self._ref_vocab_len(None, references)
        return ref_vocab_len / self._vocab_size
