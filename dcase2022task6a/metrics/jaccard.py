#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nltk.stem import SnowballStemmer
from torch import nn


class Jaccard(nn.Module):
    """Jaccard similarity, also known as "intersection over union"."""

    def __init__(self, use_stemmer: bool = True, language: str = "english") -> None:
        super().__init__()
        self.use_stemmer = use_stemmer
        self.stemmer = SnowballStemmer(language)

    def forward(
        self,
        hypotheses: list[list[str]],
        references: list[list[list[str]]],
    ) -> float:
        scores = []
        for hyp, refs in zip(hypotheses, references):
            if self.use_stemmer:
                hyp = [self.stemmer.stem(token) for token in hyp]
                refs = [[self.stemmer.stem(token) for token in ref] for ref in refs]

            hyp = set(hyp)
            refs = [set(ref) for ref in refs]

            similarities = []
            for ref in refs:
                similarity = len(hyp.intersection(ref)) / len(hyp.union(ref))
                similarities.append(similarity)

            if len(similarities) > 0:
                sim = sum(similarities) / len(similarities)
            else:
                sim = 0.0
            scores.append(sim)

        score = sum(scores) / len(scores)
        return score
