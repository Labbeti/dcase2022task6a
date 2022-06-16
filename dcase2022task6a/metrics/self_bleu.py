#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy

from typing import Optional, Union

import torch

from torch import nn

from dcase2022task6a.metrics.bleu import Bleu


class HypSelfBleu(nn.Module):
    def __init__(
        self,
        max_ngram_sizes: int = 4,
        max_refs: Optional[int] = None,
        generator: Union[None, int, torch.Generator] = 1234,
    ) -> None:
        super().__init__()
        self.max_ngram_sizes = max_ngram_sizes
        self.max_refs = max_refs
        self.generator = generator

    def forward(
        self,
        hypotheses: list[list[str]],
        references: list[list[list[str]]],
        return_dict: bool = False,
    ) -> Union[float, dict]:
        return self_bleu(
            hypotheses,
            self.max_ngram_sizes,
            self.max_refs,
            return_dict,
            generator=self.generator,
        )


class RefSelfBleu(nn.Module):
    def __init__(
        self,
        max_ngram_sizes: int = 4,
        max_refs: Optional[int] = None,
        generator: Union[None, int, torch.Generator] = 1234,
    ) -> None:
        super().__init__()
        self.max_ngram_sizes = max_ngram_sizes
        self.max_refs = max_refs
        self.generator = generator

    def forward(
        self,
        hypotheses: list[list[str]],
        references: list[list[list[str]]],
        return_dict: bool = False,
    ) -> Union[float, dict]:
        references_flat = [ref for refs in references for ref in refs]
        return self_bleu(
            references_flat,
            self.max_ngram_sizes,
            self.max_refs,
            return_dict,
            generator=self.generator,
        )


def self_bleu(
    sentences: list[list[str]],
    max_ngram_sizes: int = 4,
    max_refs: Optional[int] = None,
    return_dict: bool = False,
    dtype: torch.dtype = torch.float64,
    generator: Union[None, int, torch.Generator] = None,
) -> Union[float, dict]:
    if isinstance(generator, int):
        generator = torch.Generator(device="cpu").manual_seed(generator)
    bleu = Bleu(max_ngram_sizes)

    local_scores = []
    for i, sentence in enumerate(sentences):
        if max_refs is None:
            other_hypotheses = copy.deepcopy(sentences)
            other_hypotheses.pop(i)
        else:
            continue_ = True
            indexes = []
            while continue_:
                indexes = torch.randperm(len(sentences), generator=generator)[
                    :max_refs
                ].tolist()
                continue_ = i in indexes
            other_hypotheses = [sentences[idx] for idx in indexes]

        score = bleu([sentence], [other_hypotheses])
        local_scores.append(score)

    local_scores = torch.as_tensor(local_scores, dtype=dtype)
    global_score = local_scores.mean().item()
    if not return_dict:
        return global_score
    else:
        return {
            "score": global_score,
            "scores": local_scores,
        }
