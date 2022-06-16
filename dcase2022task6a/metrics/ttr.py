#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn


class HypTTR(nn.Module):
    """Type-Token Ratio (TTR) for hypotheses."""

    def forward(
        self, hypotheses: list[list[str]], references: list[list[list[str]]]
    ) -> float:
        scores = [ttr(hyp) for hyp in hypotheses]
        score = sum(scores) / len(scores)
        return score


class RefTTR(nn.Module):
    """Type-Token Ratio (TTR) for references"""

    def forward(
        self, hypotheses: list[list[str]], references: list[list[list[str]]]
    ) -> float:
        scores = [ttr(ref) for refs in references for ref in refs]
        if len(scores) == 0:
            return 0.0

        score = sum(scores) / len(scores)
        return score


def ttr(words: list[str]) -> float:
    if len(words) == 0:
        return 0.0
    else:
        return len(set(words)) / len(words)
