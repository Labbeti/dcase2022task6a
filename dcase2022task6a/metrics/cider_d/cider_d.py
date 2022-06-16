#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Microsoft COCO caption metric 'CIDEr'.

    IMPORTED FROM :
        - https://github.com/peteanderson80/coco-caption/blob/master/pycocoevalcap/cider/cider.py

    AUTHORS :
        - Ramakrishna Vedantam <vrama91@vt.edu>
        - Tsung-Yi Lin <tl483@cornell.edu>

    MODIFIED : Yes
        - replace iteritems -> items for python 3
        - imports
        - typing
        - fix module when n_gram != 4
"""

# fname: cider.py
#
# Description: Describes the class to compute the CIDEr (Consensus-Based Image Description Evaluation) Metric
#                by Vedantam, Zitnick, and Parikh (http://arxiv.org/abs/1411.5726)
#
# Creation Date: Sun Feb  8 14:16:54 2015
#
# Authors: Ramakrishna Vedantam <vrama91@vt.edu> and Tsung-Yi Lin <tl483@cornell.edu>

# !/usr/bin/env python
# Tsung-Yi Lin <tl483@cornell.edu>
# Ramakrishna Vedantam <vrama91@vt.edu>

__author__ = (
    "Ramakrishna Vedantam <vrama91@vt.edu> and Tsung-Yi Lin <tl483@cornell.edu>"
)

from typing import Any, Union

import numpy as np
import torch

from torch import nn

from dcase2022task6a.metrics.cider_d.scorer import CiderDScorer
from dcase2022task6a.metrics.utils import format_to_coco


class CiderD(nn.Module):
    """
    Main Class to compute the CIDEr metric
    """

    def __init__(
        self,
        ngrams_max: int = 4,
        sigma: float = 6.0,
    ) -> None:
        super().__init__()
        # set cider to sum over 1 to 4-grams
        self.ngrams_max = ngrams_max
        # set the standard deviation parameter for gaussian penalty
        self.sigma = sigma

    def forward(
        self,
        hypotheses: list[str],
        references: list[list[str]],
        return_dict: bool = False,
    ) -> Union[float, dict]:
        if len(hypotheses) != len(references):
            raise ValueError(
                f"Number of hypothesis and references are different (found {len(hypotheses)=} != {len(references)=})."
            )
        if len(hypotheses) <= 1:
            raise ValueError(
                f"CIDEr metric does not support less than 2 hypotheses with 2 lists of references. (found {len(hypotheses)=}, but expected > 1)"
            )

        res, gts = format_to_coco(hypotheses, references)
        score, scores = self.compute_score(gts, res)
        score = score.item()
        scores = torch.from_numpy(scores)

        self._last_score = score

        if not return_dict:
            return score
        else:
            return {
                "score": score,
                "scores": scores,
            }

    def get_last_score(self) -> float:
        return self._last_score

    def compute_score(
        self,
        gts: dict[Any, list[str]],
        res: dict[Any, list[str]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Main function to compute CIDEr score
        :param  hypo_for_image (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
                        ref_for_image (dict)  : dictionary with key <image> and value <tokenized reference sentence>
        :return: cider (float) : computed CIDEr score for the corpus
        """

        assert sorted(gts.keys()) == sorted(res.keys())
        imgIds = sorted(gts.keys())
        cider_scorer = CiderDScorer(n=self.ngrams_max, sigma=self.sigma)

        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert type(hypo) is list, f"{hypo=}"
            assert len(hypo) == 1, f"{hypo=}"
            assert type(ref) is list, f"{ref=}"
            assert len(ref) >= 1, f"{ref=}"

            cider_scorer += (hypo[0], ref)

        (score, scores) = cider_scorer.compute_score()

        return score, scores

    def method(self) -> str:
        return "CIDEr"
