#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re

from typing import Iterable

from dcase2022task6a.tokenization.constants import SPECIAL_TOKENS


class NormalizerI:
    def normalize_batch(self, sentences: Iterable[str]) -> list[str]:
        raise NotImplementedError("Abstract method")


class NormalizerList(NormalizerI, list):
    def __init__(self, *normalizers: NormalizerI) -> None:
        NormalizerI.__init__(self)
        list.__init__(self, normalizers)

    def normalize_batch(self, sentences: Iterable[str]) -> list[str]:
        normalized_sentences = list(sentences)
        for normalizer in self:
            normalized_sentences = normalizer.normalize_batch(normalized_sentences)
        return normalized_sentences


class Lowercase(NormalizerI):
    def normalize_batch(self, sentences: Iterable[str]) -> list[str]:
        return list(map(str.lower, sentences))


class Strip(NormalizerI):
    def normalize_batch(self, sentences: Iterable[str]) -> list[str]:
        return [sentence.strip() for sentence in sentences]


class Replace(NormalizerI):
    def __init__(self, pattern: str, repl: str) -> None:
        super().__init__()
        self._pattern = re.compile(pattern)
        self._repl = repl

    def normalize_batch(self, sentences: Iterable[str]) -> list[str]:
        return [re.sub(self._pattern, self._repl, sentence) for sentence in sentences]


class CleanDoubleSpaces(Replace):
    def __init__(self) -> None:
        super().__init__(" +", " ")


class CleanPunctuation(Replace):
    def __init__(self, pattern: str = r"[.!?;:\"“”’`\(\)\{\}\[\]\*\×,]") -> None:
        super().__init__(pattern, " ")


class CleanSpacesBeforePunctuation(Replace):
    def __init__(self) -> None:
        super().__init__(r'\s+([,.!?;:"\'](?:\s|$))', r"\1")


class CleanSpecialTokens(Replace):
    def __init__(self) -> None:
        super().__init__(f"({'|'.join(SPECIAL_TOKENS)})", "")
