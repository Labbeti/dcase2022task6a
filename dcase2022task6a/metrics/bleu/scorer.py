#!/usr/bin/env python

# bleu_scorer.py
# David Chiang <chiang@isi.edu>

# Copyright (c) 2004-2006 University of Maryland. All rights
# reserved. Do not redistribute without permission from the
# author. Not for commercial use.

# Modified by:
# Hao Fang <hfang@uw.edu>
# Tsung-Yi Lin <tl483@cornell.edu>

# =================================================================
# This code was pulled from https://github.com/tylin/coco-caption
# and refactored for Python 3.
# Image-specific names and comments have also been changed to be audio-specific
# =================================================================

"""Provides:
cook_refs(refs, n=4): Transform a list of reference sentences as strings into a form usable by cook_test().
cook_test(test, refs, n=4): Transform a test sentence as a string (together with the cooked reference sentences) into a form usable by score_cooked().
"""

import copy
import math

from collections import defaultdict
from typing import Any, Optional, Union


def precook(s: str, n: int = 4, out: bool = False) -> tuple[int, defaultdict]:
    """Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well."""
    words = s.split()
    counts = defaultdict(int)
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i : i + k])
            counts[ngram] += 1
    return (len(words), counts)


def cook_refs(
    refs: list[str], eff: Optional[str] = None, n: int = 4
) -> tuple[Any, dict]:  # lhuang: oracle will call with "average"
    """Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them."""

    reflen = []
    maxcounts = {}
    for ref in refs:
        rl, counts = precook(ref, n)
        reflen.append(rl)
        for (ngram, count) in counts.items():
            maxcounts[ngram] = max(maxcounts.get(ngram, 0), count)

    # Calculate effective reference sentence length.
    if eff == "shortest":
        reflen = min(reflen)
    elif eff == "average":
        reflen = float(sum(reflen)) / len(reflen)

    # lhuang: N.B.: leave reflen computaiton to the very end!!

    # lhuang: N.B.: in case of "closest", keep a list of reflens!! (bad design)

    return (reflen, maxcounts)


def cook_test(
    test: str, reflen_refmaxcounts: tuple, eff: Optional[None] = None, n: int = 4
) -> dict[str, Any]:
    """Takes a test sentence and returns an object that
    encapsulates everything that BLEU needs to know about it."""

    testlen, counts = precook(test, n, True)

    result = {}

    reflen, refmaxcounts = reflen_refmaxcounts  # Replaces the tuple unpacking

    # Calculate effective reference sentence length.

    if eff == "closest":
        result["reflen"] = min((abs(len - testlen), len) for len in reflen)[1]
    else:  # i.e., "average" or "shortest" or None
        result["reflen"] = reflen

    result["testlen"] = testlen

    result["guess"] = [max(0, testlen - k + 1) for k in range(1, n + 1)]

    result["correct"] = [0] * n
    for (ngram, count) in counts.items():
        result["correct"][len(ngram) - 1] += min(refmaxcounts.get(ngram, 0), count)

    return result


class BleuScorer:
    """Bleu scorer."""

    __slots__ = (
        "n",
        "crefs",
        "ctest",
        "_score",
        "_ratio",
        "_testlen",
        "_reflen",
        "special_reflen",
    )
    # special_reflen is used in oracle (proportional effective ref len for a node).

    def copy(self) -> "BleuScorer":
        """copy the refs."""
        new = BleuScorer(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        new._score = None
        return new

    def __init__(self, test=None, refs=None, n=4, special_reflen=None):
        """singular instance"""

        self.n = n
        self.crefs = []
        self.ctest = []
        self.cook_append(test, refs)
        self.special_reflen = special_reflen

    def cook_append(self, test: Optional[str], refs: Optional[list[str]]) -> None:
        """called by constructor and __iadd__ to avoid creating new instances."""

        if refs is not None:
            self.crefs.append(cook_refs(refs))
            if test is not None:
                cooked_test = cook_test(test, self.crefs[-1])
                self.ctest.append(cooked_test)  # N.B.: -1
            else:
                self.ctest.append(None)  # lens of crefs and ctest have to match

        self._score = None  # need to recompute

    def ratio(self, option: Optional[str] = None) -> Any:
        self.compute_score(option=option)
        return self._ratio

    def reflen(self, option: Optional[str] = None) -> int:
        self.compute_score(option=option)
        return self._reflen

    def testlen(self, option: Optional[str] = None) -> int:
        self.compute_score(option=option)
        return self._testlen

    def retest(self, new_test: Union[str, list[str]]) -> "BleuScorer":
        if type(new_test) is str:
            new_test = [new_test]
        assert len(new_test) == len(self.crefs), new_test
        self.ctest = []
        for t, rs in zip(new_test, self.crefs):
            self.ctest.append(cook_test(t, rs))
        self._score = None

        return self

    def rescore(self, new_test: Union[str, list[str]]) -> tuple:
        """replace test(s) with new test(s), and returns the new score."""
        return self.retest(new_test).compute_score()

    def size(self) -> int:
        assert len(self.crefs) == len(self.ctest), "refs/test mismatch! %d<>%d" % (
            len(self.crefs),
            len(self.ctest),
        )
        return len(self.crefs)

    def __iadd__(self, other: Union[tuple, "BleuScorer"]) -> "BleuScorer":
        """add an instance (e.g., from another sentence)."""

        if isinstance(other, tuple):
            # avoid creating new BleuScorer instances
            self.cook_append(other[0], other[1])
        else:
            assert self.compatible(other), "incompatible BLEUs."
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)
            self._score = None  # need to recompute

        return self

    def compatible(self, other: Any) -> bool:
        return isinstance(other, BleuScorer) and self.n == other.n

    def single_reflen(self, option: Optional[str] = "average") -> float:
        return self._single_reflen(self.crefs[0][0], option)

    def _single_reflen(
        self,
        reflens: list[int],
        option: Optional[str] = None,
        testlen: Optional[int] = None,
    ) -> float:
        if option == "shortest":
            reflen = min(reflens)
        elif option == "average":
            reflen = float(sum(reflens)) / len(reflens)
        elif option == "closest":
            assert testlen is not None
            reflen = min((abs(len - testlen), len) for len in reflens)[1]
        else:
            assert False, "unsupported reflen option %s" % option

        return reflen

    def recompute_score(
        self, option: Optional[str] = None, verbose: int = 0
    ) -> tuple[list[float], list[list[float]]]:
        self._score = None
        return self.compute_score(option, verbose)

    def compute_score(
        self, option: Optional[str] = None, verbose: int = 0
    ) -> tuple[list[float], list[list[float]]]:
        n = self.n
        small = 1e-9
        tiny = 1e-15  # so that if guess is 0 still return 0
        bleu_list = [[] for _ in range(n)]

        if self._score is not None:
            return self._score, []

        if option is None:
            option = "average" if len(self.crefs) == 1 else "closest"

        self._testlen = 0
        self._reflen = 0
        totalcomps = {"testlen": 0, "reflen": 0, "guess": [0] * n, "correct": [0] * n}

        # for each sentence
        for comps in self.ctest:
            testlen = comps["testlen"]
            self._testlen += testlen

            if self.special_reflen is None:  # need computation
                reflen = self._single_reflen(comps["reflen"], option, testlen)
            else:
                reflen = self.special_reflen

            self._reflen += reflen

            for key in ["guess", "correct"]:
                for k in range(n):
                    totalcomps[key][k] += comps[key][k]

            # append per audio bleu score
            bleu = 1.0
            for k in range(n):
                bleu *= (float(comps["correct"][k]) + tiny) / (
                    float(comps["guess"][k]) + small
                )
                bleu_list[k].append(bleu ** (1.0 / (k + 1)))
            ratio = (testlen + tiny) / (reflen + small)  # N.B.: avoid zero division
            if ratio < 1:
                for k in range(n):
                    bleu_list[k][-1] *= math.exp(1 - 1 / ratio)

            if verbose > 1:
                print(comps, reflen)

        totalcomps["reflen"] = self._reflen
        totalcomps["testlen"] = self._testlen

        bleus = []
        bleu = 1.0
        for k in range(n):
            bleu *= float(totalcomps["correct"][k] + tiny) / (
                totalcomps["guess"][k] + small
            )
            bleus.append(bleu ** (1.0 / (k + 1)))
        ratio = (self._testlen + tiny) / (
            self._reflen + small
        )  # N.B.: avoid zero division
        if ratio < 1:
            for k in range(n):
                bleus[k] *= math.exp(1 - 1 / ratio)

        if verbose > 0:
            print(totalcomps)
            print("ratio:", ratio)

        self._score = bleus
        return self._score, bleu_list
