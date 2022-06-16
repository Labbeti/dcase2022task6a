#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Callable, Iterable, Optional, Union

import torch

from dcase2022task6a.metrics.bert_score import BertScore
from dcase2022task6a.metrics.bleu import Bleu
from dcase2022task6a.metrics.cider_d import CiderD
from dcase2022task6a.metrics.diversity import (
    HypDiversity,
    RefDiversity,
    HypOnRefDiversity,
)
from dcase2022task6a.metrics.jaccard import Jaccard
from dcase2022task6a.metrics.meteor import Meteor
from dcase2022task6a.metrics.rouge_l import RougeL
from dcase2022task6a.metrics.spice import Spice
from dcase2022task6a.metrics.stats import (
    GlobalVocabUsage,
    GlobalVocabCoverage,
    GlobalVocabFreq,
    HypMeanLen,
    RefMeanLen,
    HypVocabLen,
    RefVocabLen,
    GlobalVocabPrecision,
    LocalVocabPrecision,
)
from dcase2022task6a.metrics.ttr import HypTTR, RefTTR
from dcase2022task6a.metrics.utils import check_sentence_level, check_word_level
from dcase2022task6a.nn.modules.misc import ParallelDict


DEFAULT_AAC_METRICS = (
    "bleu_1",
    "bleu_2",
    "bleu_3",
    "bleu_4",
    "meteor",
    "rouge_l",
    "cider_d",
    "spice",
    "spider",
)


class WordLevelMetrics(ParallelDict):
    def __init__(
        self,
        include: Optional[Iterable[str]] = None,
        exclude: Iterable[str] = (),
        check_input: bool = False,
        verbose: int = 0,
    ) -> None:
        include = dict.fromkeys(include) if include is not None else None
        exclude = dict.fromkeys(exclude)
        metrics_factories = self.get_metrics_factories()
        metrics = {
            name: factory()
            for name, factory in metrics_factories.items()
            if (name not in exclude) and (include is None or name in include)
        }
        super().__init__(metrics, verbose >= 1)
        self._check_input = check_input

    @classmethod
    def get_metrics_factories(cls) -> dict[str, Callable]:
        return {
            "global_vocab_usage": GlobalVocabUsage,
            "global_vocab_coverage": GlobalVocabCoverage,
            "global_vocab_freq": GlobalVocabFreq,
            "global_vocab_precision": GlobalVocabPrecision,
            "local_vocab_precision": LocalVocabPrecision,
            "hyp_mean_len": HypMeanLen,
            "ref_mean_len": RefMeanLen,
            "hyp_words_set_len": HypVocabLen,
            "ref_words_set_len": RefVocabLen,
            "hyp_ttr": HypTTR,
            "ref_ttr": RefTTR,
            "jaccard": Jaccard,
            "hyp_diversity4": lambda: HypDiversity(4),
            "ref_diversity4": lambda: RefDiversity(4),
            "hr_diversity4": lambda: HypOnRefDiversity(4),
        }

    def forward(self, *args, **kwargs) -> dict[str, Any]:
        if self._check_input:
            valid = check_word_level(*args, **kwargs)
            if not valid:
                raise ValueError(f"Invalid input for {self.__class__.__name__}.")
        return super().forward(*args, **kwargs)


class SentenceLevelMetrics(ParallelDict):
    def __init__(
        self,
        java_path: str = "java",
        ext_dpath: str = "ext",
        tmp_dpath: str = "/tmp",
        device: Union[torch.device, str, None] = "cpu",
        include: Optional[Iterable[str]] = None,
        exclude: Iterable[str] = (),
        check_input: bool = False,
        verbose: int = 0,
    ) -> None:
        include = dict.fromkeys(include) if include is not None else None
        exclude = dict.fromkeys(exclude)

        if include is not None and "spider" in include:
            include |= dict.fromkeys(("spice", "cider"))

        metrics_factories = self.get_metrics_factories(
            java_path, ext_dpath, tmp_dpath, device, verbose
        )
        metrics = {
            name: factory()
            for name, factory in metrics_factories.items()
            if (name not in exclude) and (include is None or name in include)
        }
        super().__init__(metrics, verbose >= 1)
        self._check_input = check_input

    @classmethod
    def get_metrics_factories(
        cls,
        java_path: str,
        ext_dpath: str,
        tmp_dpath: str,
        device: Union[torch.device, str, None],
        verbose: int = 0,
    ) -> dict[str, Callable]:
        return {
            "bert_score": lambda: BertScore(device=device),
            "bleu_1": lambda: Bleu(1),
            "bleu_2": lambda: Bleu(2),
            "bleu_3": lambda: Bleu(3),
            "bleu_4": lambda: Bleu(4),
            "meteor": lambda: Meteor(
                java_path=java_path,
                ext_dpath=ext_dpath,
                verbose=verbose,
            ),
            "rouge_l": lambda: RougeL(),
            "cider_d": lambda: CiderD(),
            "spice": lambda: Spice(
                java_path=java_path,
                ext_dpath=ext_dpath,
                tmp_dpath=tmp_dpath,
                verbose=verbose >= 2,
            ),
        }

    def forward(self, *args, **kwargs) -> dict[str, Any]:
        if self._check_input:
            valid = check_sentence_level(*args, **kwargs)
            if not valid:
                raise ValueError(f"Invalid input for {self.__class__.__name__}.")

        scores = super().forward(*args, **kwargs)

        if "cider_d" in scores.keys() and "spice" in scores.keys():
            cider_scores = scores["cider_d"]
            spice_scores = scores["spice"]
            if isinstance(cider_scores, dict) and isinstance(spice_scores, dict):
                keys = {"score", "scores"}
                assert set(cider_scores.keys()) == set(spice_scores.keys()) == keys
                scores["spider"] = {
                    key: (cider_scores[key] + spice_scores[key]) / 2.0 for key in keys
                }
            else:
                scores["spider"] = (cider_scores + spice_scores) / 2.0
        return scores
