#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Union

import torch

from bert_score.scorer import BERTScorer
from torch import nn


class BertScore(nn.Module):
    """
    The string-level input is "sentence".
    """

    def __init__(
        self,
        device: Union[torch.device, str, None] = None,
        dtype: Optional[torch.dtype] = torch.float64,
    ) -> None:
        super().__init__()
        self.scorer = BERTScorer(
            lang="en",
            model_type="roberta-large",
            num_layers=17,
            device=device,
        )
        self.dtype = dtype

    def forward(
        self,
        hypotheses: list[str],
        references: list[list[str]],
        return_dict: bool = False,
    ) -> Union[float, dict]:
        if any(len(refs) == 0 for refs in references):
            raise RuntimeError(f"Found an empty reference for BertScore.")

        precisions, recalls, fscores = self.scorer.score(hypotheses, references)
        scores = torch.as_tensor(fscores, dtype=self.dtype)  # type: ignore
        score = scores.mean().item()

        if not return_dict:
            return score
        else:
            return {
                "score": score,
                "scores": scores,
                "precisions": precisions,
                "recalls": recalls,
            }
