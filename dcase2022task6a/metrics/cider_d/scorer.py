#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Microsoft COCO caption metric 'CIDEr'.

    IMPORTED FROM :
        - https://github.com/peteanderson80/coco-caption/blob/master/pycocoevalcap/cider/cider_scorer.py
"""

import copy
import math

from collections import defaultdict
from typing import Optional

import numpy as np


class CiderDScorer:
    """CIDEr scorer."""

    def __init__(
        self,
        test: Optional[str] = None,
        refs: Optional[list[str]] = None,
        n: int = 4,
        sigma: float = 6.0,
    ) -> None:
        """singular instance"""
        super().__init__()
        self.n = n
        self.sigma = sigma
        self.crefs = []
        self.ctest = []
        self.document_frequency = defaultdict(float)
        self.cook_append(test, refs)
        self.log_ref_len = None

    def copy(self) -> "CiderDScorer":
        """copy the refs."""
        new = CiderDScorer(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        return new

    def cook_append(self, test: Optional[str], refs: Optional[list[str]]) -> None:
        """called by constructor and __iadd__ to avoid creating new instances."""

        if refs is not None:
            self.crefs.append(cook_refs(refs, self.n))
            if test is not None:
                self.ctest.append(cook_test(test, self.n))  # N.B.: -1
            else:
                self.ctest.append(None)  # lens of crefs and ctest have to match

    def size(self) -> int:
        assert len(self.crefs) == len(self.ctest), "refs/test mismatch! %d<>%d" % (
            len(self.crefs),
            len(self.ctest),
        )
        return len(self.crefs)

    def __iadd__(self, other) -> "CiderDScorer":
        """add an instance (e.g., from another sentence)."""

        if type(other) is tuple:
            # avoid creating new CiderDScorer instances
            self.cook_append(other[0], other[1])
        else:
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)

        return self

    def get_document_freqs(self) -> dict:
        return self.document_frequency

    def compute_doc_freq(self) -> None:
        """
        Compute term frequency for reference data.
        This will be used to compute idf (inverse document frequency later)
        The term frequency is stored in the object
        :return: None
        """
        for refs in self.crefs:
            # refs, k ref captions of one image
            for ngram in set([ngram for ref in refs for ngram, count in ref.items()]):
                self.document_frequency[ngram] += 1
        # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)

        # compute log reference length
        self.log_ref_len = np.log(float(len(self.crefs)))

    def compute_cider(self) -> list[float]:
        def counts2vec(cnts: dict) -> tuple:
            """
            Function maps counts of ngram to vector of tfidf weights.
            The function returns vec, an array of dictionary that store mapping of n-gram and tf-idf weights.
            The n-th entry of array denotes length of n-grams.
            :param cnts:
            :return: vec (array of dict), norm (array of float), length (int)
            """
            assert self.log_ref_len is not None
            vec = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]
            for (ngram, term_freq) in cnts.items():
                # give word count 1 if it doesn't appear in reference corpus
                log_df = np.log(max(1.0, self.document_frequency[ngram]))

                # ngram index
                n = len(ngram) - 1

                # tf (term_freq) * idf (precomputed idf) for n-grams
                vec[n][ngram] = float(term_freq) * (self.log_ref_len - log_df)

                # compute norm for the vector.  the norm will be used for computing similarity
                norm[n] += pow(vec[n][ngram], 2)

                if n == 1:
                    length += term_freq
            norm = [np.sqrt(n) for n in norm]
            return vec, norm, length

        def sim(
            vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref
        ) -> np.ndarray:
            """
            Compute the cosine similarity of two vectors.
            :param vec_hyp: array of dictionary for vector corresponding to hypothesis
            :param vec_ref: array of dictionary for vector corresponding to reference
            :param norm_hyp: array of float for vector corresponding to hypothesis
            :param norm_ref: array of float for vector corresponding to reference
            :param length_hyp: int containing length of hypothesis
            :param length_ref: int containing length of reference
            :return: array of score for each n-grams cosine similarity
            """
            delta = float(length_hyp - length_ref)
            # measure consine similarity
            val = np.array([0.0 for _ in range(self.n)])
            for n in range(self.n):
                # ngram
                for (ngram, count) in vec_hyp[n].items():
                    # vrama91 : added clipping
                    val[n] += (
                        min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]
                    )

                if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
                    val[n] /= norm_hyp[n] * norm_ref[n]

                assert not math.isnan(val[n])
                # vrama91: added a length based gaussian penalty
                val[n] *= np.e ** (-(delta ** 2) / (2 * self.sigma ** 2))
            return val

        scores = []
        for test, refs in zip(self.ctest, self.crefs):
            # compute vector for test captions
            vec, norm, length = counts2vec(test)
            # compute vector for ref captions
            score = np.array([0.0 for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = counts2vec(ref)
                score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            # change by vrama91 - mean of ngram scores, instead of sum
            score_avg = np.mean(score)
            # divide by number of references
            score_avg /= len(refs)
            # multiply score by 10
            score_avg *= 10.0
            # append score of an image to the score list
            scores.append(score_avg)
        return scores

    def compute_score(self) -> tuple[np.ndarray, np.ndarray]:
        # compute idf
        self.compute_doc_freq()
        # assert to check document frequency
        assert len(self.ctest) >= max(self.document_frequency.values())
        # compute cider score
        scores = self.compute_cider()
        scores = np.array(scores)
        return np.mean(scores), scores


def precook(s: str, n: int) -> defaultdict:
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    words = s.split()
    counts = defaultdict(int)
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i : i + k])
            counts[ngram] += 1
    return counts


def cook_refs(
    refs: list[str], n: int
) -> list[defaultdict]:  # lhuang: oracle will call with 'average'
    """Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    """
    return [precook(ref, n) for ref in refs]


def cook_test(test: str, n: int) -> defaultdict:
    """Takes a test sentence and returns an object that
    encapsulates everything that BLEU needs to know about it.
    :param test: list of string : hypothesis sentence for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (dict)
    """
    return precook(test, n)
