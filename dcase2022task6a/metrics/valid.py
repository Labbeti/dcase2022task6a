#!/usr/bin/env python
# -*- coding: utf-8 -*-


from collections import Counter
from functools import cache

from nltk.corpus import words
from torch import nn


@cache
def get_vocab(source: str) -> dict[str, None]:
    if source == "nltk":
        vocab = dict.fromkeys(words.words())
    else:
        raise ValueError(f"Invalid argument {source=}.")
    return vocab


class HypValidVocab(nn.Module):
    def __init__(self, source: str = "nltk") -> None:
        super().__init__()
        self._vocab = get_vocab(source)

    def forward(
        self,
        hypotheses: list[list[str]],
        references: list[list[list[str]]],
    ) -> float:
        vocab = Counter(word for hyp in hypotheses for word in hyp)
        valid_vocab = [word for word in vocab.keys() if word in self._vocab]
        return len(valid_vocab) / len(vocab)


class RefValidVocab(nn.Module):
    def __init__(self, source: str = "nltk") -> None:
        super().__init__()
        self._vocab = get_vocab(source)

    def forward(
        self,
        hypotheses: list[list[str]],
        references: list[list[list[str]]],
    ) -> float:
        vocab = Counter(word for refs in references for ref in refs for word in ref)
        valid_vocab = [word for word in vocab.keys() if word in self._vocab]
        return len(valid_vocab) / len(vocab)
