#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Description : Computes ROUGE-L metric as described by Lin and Hovey (2004)
#
# Creation Date : 2015-01-07 06:03
# Author : Ramakrishna Vedantam <vrama91@vt.edu>

# =================================================================
# This code was pulled from https://github.com/tylin/coco-caption
# Image-specific names and comments have been changed to be audio-specific
# =================================================================

from typing import Union

import numpy as np

import torch

from torch import nn

from dcase2022task6a.metrics.utils import format_to_coco


class RougeL(nn.Module):
    """
    Class for computing ROUGE-L score for a set of candidate sentences for the MS COCO test set
    """

    def __init__(self, beta: float = 1.2) -> None:
        # vrama91: updated the value below based on discussion with Hovey
        super().__init__()
        self.beta = beta

    def forward(
        self,
        hypotheses: list[str],
        references: list[list[str]],
        return_dict: bool = False,
    ) -> Union[float, dict]:
        res, gts = format_to_coco(hypotheses, references)
        score, scores = self.compute_score(gts, res)
        score = score.item()
        scores = torch.from_numpy(scores)

        if not return_dict:
            return score
        else:
            return {
                "score": score,
                "scores": scores,
            }

    def compute_score(self, gts: dict, res: dict) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes Rouge-L score given a set of reference and candidate sentences for the dataset
        Invoked by evaluate_captions.py
        :param hypo_for_audio: dict : candidate / test sentences with "audio name" key and "tokenized sentences" as values
        :param ref_for_audio: dict : reference captions with "audio name" key and "tokenized sentences" as values
        :returns: average_score: float (mean ROUGE-L score computed by averaging scores for all the audio files)
        """
        assert gts.keys() == res.keys()
        audioIds = gts.keys()

        scores = []
        for id in audioIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert type(hypo) is list
            assert len(hypo) == 1
            assert type(ref) is list
            assert len(ref) > 0

            scores.append(self.calc_score(hypo, ref))

        score = np.mean(np.array(scores))
        return score, np.array(scores)

    def calc_score(self, candidate: list[str], refs: list[str]) -> float:
        """
        Compute ROUGE-L score given one candidate and references for an audio
        :param candidate: str : candidate sentence to be evaluated
        :param refs: list of str : COCO reference sentences for the particular audio to be evaluated
        :returns score: int (ROUGE-L score for the candidate evaluated against references)
        """
        assert len(candidate) == 1
        assert len(refs) > 0
        prec = []
        rec = []

        # split into tokens
        token_c = candidate[0].split(" ")

        for reference in refs:
            # split into tokens
            token_r = reference.split(" ")
            # compute the longest common subsequence
            lcs = my_lcs(token_r, token_c)
            prec.append(lcs / float(len(token_c)))
            rec.append(lcs / float(len(token_r)))

        prec_max = max(prec)
        rec_max = max(rec)

        if prec_max != 0 and rec_max != 0:
            score = ((1 + self.beta ** 2) * prec_max * rec_max) / float(
                rec_max + self.beta ** 2 * prec_max
            )
        else:
            score = 0.0
        return score

    def method(self):
        return "Rouge"


def my_lcs(string: list[str], sub: list[str]) -> int:
    """
    Calculates longest common subsequence for a pair of tokenized strings
    :param string : list of str : tokens from a string split using whitespace
    :param sub : list of str : shorter string, also split using whitespace
    :returns: length (list of int): length of the longest common subsequence between the two strings
    Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
    """
    if len(string) < len(sub):
        sub, string = string, sub

    lengths = [[0 for i in range(0, len(sub) + 1)] for j in range(0, len(string) + 1)]

    for j in range(1, len(sub) + 1):
        for i in range(1, len(string) + 1):
            if string[i - 1] == sub[j - 1]:
                lengths[i][j] = lengths[i - 1][j - 1] + 1
            else:
                lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])

    return lengths[len(string)][len(sub)]
