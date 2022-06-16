#!/usr/bin/env python
#
# File Name : bleu.py
#
# Description : Wrapper for BLEU scorer.
#
# Creation Date : 06-01-2015
# Last Modified : Thu 19 Mar 2015 09:13:28 PM PDT
# Authors : Hao Fang <hfang@uw.edu> and Tsung-Yi Lin <tl483@cornell.edu>

# =================================================================
# This code was pulled from https://github.com/tylin/coco-caption
# and refactored for Python 3.
# Image-specific names and comments have also been changed to be audio-specific
# =================================================================

from typing import Union

import torch

from torch import nn

from dcase2022task6a.metrics.bleu.scorer import BleuScorer
from dcase2022task6a.metrics.utils import format_to_coco


class Bleu(nn.Module):
    def __init__(
        self,
        max_ngram_sizes: int = 4,
        option: str = "closest",
        verbose: int = 0,
    ) -> None:
        """Default compute Blue score up to 4."""
        OPTIONS = ("shortest", "average", "closest")
        if option not in OPTIONS:
            raise ValueError(f"Invalid option {option=}. (expected one of {OPTIONS})")

        super().__init__()
        self.max_ngram_sizes = max_ngram_sizes
        self.option = option
        self.verbose = verbose
        self.dtype = torch.float64

    def forward(
        self,
        hypotheses: list[str],
        references: list[list[str]],
        return_dict: bool = False,
    ) -> Union[float, dict]:
        res, gts = format_to_coco(hypotheses, references)
        score_1_to_n, scores_1_to_n = self.compute_score(gts, res)

        score_n = score_1_to_n[-1]
        scores_n = torch.as_tensor(scores_1_to_n[-1], dtype=self.dtype)
        score_1_to_n = torch.as_tensor(score_1_to_n, dtype=self.dtype)
        scores_1_to_n = torch.as_tensor(scores_1_to_n, dtype=self.dtype)

        if not return_dict:
            return score_n
        else:
            return {
                "score": score_n,
                "scores": scores_n,
                "score_1_to_n": score_1_to_n,
                "scores_1_to_n": scores_1_to_n,
            }

    def compute_score(
        self, gts: dict, res: dict
    ) -> tuple[list[float], list[list[float]]]:
        assert gts.keys() == res.keys()
        audioIds = gts.keys()

        bleu_scorer = BleuScorer(n=self.max_ngram_sizes)
        for id in audioIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert type(hypo) is list
            assert len(hypo) == 1
            assert type(ref) is list
            assert len(ref) >= 1

            bleu_scorer += (hypo[0], ref)

        score, scores = bleu_scorer.compute_score(
            option=self.option, verbose=self.verbose
        )
        return score, scores

    def method(self) -> str:
        return "Bleu"
