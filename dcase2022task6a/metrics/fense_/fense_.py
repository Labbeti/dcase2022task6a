#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from typing import Optional

from torchmetrics import Metric

from dcase2022task6a.metrics.fense_.evaluator import Evaluator


class Fense(Metric):
    """Fluency ENhanced Sentence-bert Evaluation (FENSE)

    Internal implementation : https://github.com/blmoistawinde/fense
    Reference : https://arxiv.org/abs/2110.04684
    """

    def __init__(
        self,
        batch_size: int = 32,
        device: str = "cuda",
        sbert_model: str = "paraphrase-TinyBERT-L6-v2",
        echecker_model: str = "none",
        error_threshold: float = 0.9,
        penalty: float = 0.9,
        use_proxy: bool = False,
        proxies: Optional[dict[str, str]] = None,
        verbose: int = 0,
    ) -> None:
        super().__init__()
        self.evaluator = Evaluator(
            batch_size,
            device,
            sbert_model,
            echecker_model,
            error_threshold,
            penalty,
            use_proxy,
            proxies,
            verbose,
        )
        self.add_state("hypotheses", [], dist_reduce_fx=self._dist_reduce_text)
        self.add_state("references", [], dist_reduce_fx=self._dist_reduce_text)

    def update(self, hypotheses: list[str], references: list[list[str]]) -> None:
        self.hypotheses += hypotheses  # type: ignore
        self.references += references  # type: ignore

    def compute(self) -> float:
        return fense_score(self.hypotheses, self.references, self.evaluator)  # type: ignore

    def _dist_reduce_text(self, sentences: list[str]) -> list[str]:
        raise NotImplementedError


def fense_score(
    hypotheses: list[str],
    references: list[list[str]],
    evaluator: Optional[Evaluator] = None,
    verbose: int = 0,
    **evaluator_kwargs,
) -> float:
    if evaluator is None:
        evaluator = Evaluator(**evaluator_kwargs, verbose=verbose)
    elif len(evaluator_kwargs) > 0:
        if verbose >= 0:
            logging.warning(
                f"Ignore kwargs {tuple(evaluator_kwargs.keys())} because Evaluator is not None."
            )

    score = evaluator.corpus_score(hypotheses, references)
    score = score.item()  # type: ignore
    return score


def test() -> None:
    # TODO : rem
    hyps = [["a", "a"], ["c", "d"]]
    refs = [[["a", "aa"], ["b"]], [["c"], ["d"]]]

    flat_hyps = hyps[0] + hyps[1]
    flat_refs = refs[0] + refs[1]
    score_1 = fense_score(
        flat_hyps, flat_refs, Evaluator(echecker_model="none", verbose=0)
    )
    print(f"{score_1=}")

    fense = Fense(verbose=0)
    fense.update(hyps[0], refs[0])
    fense.update(hyps[1], refs[1])
    score_2 = fense.compute()
    print(f"{score_2=}")

    print(f"EQUAL: {score_1 == score_2}")


if __name__ == "__main__":
    test()
