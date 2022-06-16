#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Optional, Union

from torch import nn

from dcase2022task6a.metrics.cider_d import CiderD
from dcase2022task6a.metrics.spice import Spice


class Spider(nn.Module):
    """
    Compute Spider score from cider and spice last scores.
    Useful for avoid to compute the slow CiderD metric twice.

    Output values are in range [0, 5.5]. Higher is better.

    Default: Spider = (CiderD + Spice) / 2
    """

    def __init__(
        self,
        cider_weight: float = 0.5,
        spice_weight: float = 0.5,
        cider_kwargs: Optional[dict[str, Any]] = None,
        spice_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        if cider_kwargs is None:
            cider_kwargs = {}
        if spice_kwargs is None:
            spice_kwargs = {}

        super().__init__()
        self._cider = CiderD(**cider_kwargs)
        self._spice = Spice(**spice_kwargs)
        self._cider_weight = cider_weight
        self._spice_weight = spice_weight

    def forward(
        self,
        hypotheses: list[str],
        references: list[list[str]],
        return_dict: bool = False,
    ) -> Union[float, dict]:
        if len(hypotheses) != len(references):
            raise ValueError(
                f"Number of hypothesis and references are different ({len(hypotheses)} != {len(references)})."
            )

        cider_outs = self._cider(hypotheses, references, return_dict=return_dict)
        spice_outs = self._spice(hypotheses, references, return_dict=return_dict)

        if not return_dict:
            score = self._cider_weight * cider_outs + self._spice_weight * spice_outs
            return score
        else:
            assert isinstance(cider_outs, dict) and isinstance(spice_outs, dict)
            assert set(cider_outs.keys()) == set(spice_outs.keys())
            outs = {}
            for key, cider_out in cider_outs.items():
                spice_out = spice_outs[key]
                outs[key] = (
                    self._cider_weight * cider_out + self._spice_weight * spice_out
                )
                outs[f"cider_{key}"] = cider_out
                outs[f"spice_{key}"] = spice_out
            return outs
