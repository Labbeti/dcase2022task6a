#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter
from typing import Union

import torch

from nltk.util import ngrams
from torch import nn


class HypDiversity(nn.Module):
    def __init__(self, max_ngram_size: int = 4) -> None:
        super().__init__()
        self.max_ngram_size = max_ngram_size

    def forward(
        self,
        hypotheses: list[list[str]],
        references: list[list[list[str]]],
        return_dict: bool = False,
    ) -> Union[float, dict]:
        hypotheses_unflat = [[hyp] for hyp in hypotheses]
        return diversity(hypotheses_unflat, self.max_ngram_size, return_dict)


class RefDiversity(nn.Module):
    def __init__(self, max_ngram_size: int = 4) -> None:
        super().__init__()
        self.max_ngram_size = max_ngram_size

    def forward(
        self,
        hypotheses: list[list[str]],
        references: list[list[list[str]]],
        return_dict: bool = False,
    ) -> Union[float, dict]:
        return diversity(references, self.max_ngram_size, return_dict)


class HypOnRefDiversity(nn.Module):
    def __init__(self, max_ngram_size: int = 4) -> None:
        super().__init__()
        self.max_ngram_size = max_ngram_size

    def forward(
        self,
        hypotheses: list[list[str]],
        references: list[list[list[str]]],
        return_dict: bool = False,
    ) -> Union[float, dict]:
        hypotheses_unflat = [[hyp] for hyp in hypotheses]
        hyp_outs = diversity(hypotheses_unflat, self.max_ngram_size, return_dict)
        ref_outs = diversity(references, self.max_ngram_size, return_dict)

        if isinstance(hyp_outs, float) and isinstance(ref_outs, float):
            return hyp_outs / ref_outs
        elif isinstance(hyp_outs, dict) and isinstance(ref_outs, dict):
            hyp_on_ref_score = hyp_outs["score"] / ref_outs["score"]
            # note : no "scores" key here because 'hyp_on_ref_score != (hyp_outs["scores"] / ref_outs["scores"]).mean()'
            outs = {
                "score": hyp_on_ref_score,
            }
            outs |= {f"hyp_{k}": v for k, v in hyp_outs.items()}
            outs |= {f"ref_{k}": v for k, v in ref_outs.items()}
            return outs
        else:
            raise TypeError(
                f"Internal error: invalid types {type(hyp_outs)=} and {type(ref_outs)=}. (expected (float, float) or (dict, dict))"
            )


class Diversity(nn.Module):
    def __init__(self, max_ngram_size: int = 4) -> None:
        super().__init__()
        self.max_ngram_size = max_ngram_size

    def forward(
        self,
        sentences: list[list],
        return_dict: bool = False,
    ) -> Union[float, dict]:
        if isinstance(sentences[0][0], str):
            sentences = [[sent] for sent in sentences]
        return diversity(sentences, self.max_ngram_size, return_dict)


def diversity(
    sentences: list[list[list[str]]],
    max_ngram_size: int,
    return_dict: bool = False,
    dtype: torch.dtype = torch.float64,
) -> Union[float, dict]:
    factory_kwargs = dict(dtype=dtype)
    full_scores = torch.empty((max_ngram_size, len(sentences)), **factory_kwargs)

    for ngram_size in range(1, max_ngram_size + 1):
        for i, sents in enumerate(sentences):
            sents_scores = torch.empty((len(sents),), **factory_kwargs)
            for j, sent in enumerate(sents):
                sent_ngrams = list(ngrams(sent, ngram_size))
                if len(sent_ngrams) == 0:
                    sent_score = 0
                else:
                    sent_score = len(set(sent_ngrams)) / len(sent_ngrams)
                sents_scores[j] = sent_score
            full_scores[ngram_size - 1, i] = sents_scores.mean()

    scores = full_scores.mean(dim=0)
    score = scores.mean().item()

    if not return_dict:
        return score
    else:
        return {
            "score": score,
            "scores": scores,
        }


# TODO : rem ?
def _diversity_old(
    sentences: list[list[str]],
    max_ngram_size: int,
    return_dict: bool = False,
    dtype: torch.dtype = torch.float64,
) -> Union[float, dict]:
    """Compute diversity score, a unweighted average of unique n-grams on total number of n-grams."""
    all_scores = []
    for ngram_size in range(1, max_ngram_size + 1):
        sentences_ngrams = [
            list(ngrams(sentence, ngram_size)) for sentence in sentences
        ]
        sentences_ngrams_counters = [
            Counter(ngrams_lst) for ngrams_lst in sentences_ngrams
        ]
        scores = [
            (len(counter) / len(ngrams_lst)) if len(ngrams_lst) > 0 else 0
            for ngrams_lst, counter in zip(sentences_ngrams, sentences_ngrams_counters)
        ]
        all_scores.append(scores)

    all_scores = torch.as_tensor(all_scores, dtype=dtype)
    # all_scores: (len(sentences), max_ngram_size)
    scores = all_scores.mean(dim=0)
    score = scores.mean().item()

    if not return_dict:
        return score
    else:
        return {
            "score": score,
            "scores": scores,
        }


# TODO : rem ?
def _global_diversity_old(
    sentences: list[list[str]],
    max_ngram_size: int,
    dtype: torch.dtype = torch.float64,
) -> float:
    all_scores = []
    for ngram_size in range(1, max_ngram_size + 1):
        sentences_ngrams = [
            list(ngrams(sentence, ngram_size)) for sentence in sentences
        ]
        all_ngrams = [
            ngram_tup
            for sentence_ngrams in sentences_ngrams
            for ngram_tup in sentence_ngrams
        ]
        n_ngrams = len(all_ngrams)
        n_unique_ngrams = len(Counter(all_ngrams))
        score = n_ngrams / n_unique_ngrams
        all_scores.append(score)

    all_scores = torch.as_tensor(all_scores, dtype=dtype)
    score = all_scores.mean().item()
    return score
