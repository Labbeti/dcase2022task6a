#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random

from typing import Any, Iterable, Union

import torch

from torch import nn


class SelectCaption(nn.Module):
    def __init__(
        self,
        mode: Union[str, int, Iterable[int]] = "random",
        keep_list: bool = False,
    ) -> None:
        """
        :param mode: 'random', 'all', int, Sequence[int].
        :param keep_list: If true, will keep the list of captions if mode provide a single caption.
        """
        if isinstance(mode, Iterable) and not isinstance(mode, str):
            mode = list(mode)

        if (
            mode not in ("random", "all")
            and not isinstance(mode, int)
            and not (
                isinstance(mode, Iterable) and all(isinstance(idx, int) for idx in mode)
            )
        ):
            raise ValueError(
                f"Invalid argument {mode=}. (expected 'random', 'all', int or Iterable of ints)"
            )

        super().__init__()
        self._mode = mode
        self._keep_list = keep_list

    def forward(
        self,
        captions: Iterable[Iterable[str]],
    ) -> Union[Iterable[str], list[Iterable[str]]]:
        captions = list(captions)

        if self._mode == "random":
            index = int(torch.randint(0, len(captions), ()).item())
        elif self._mode == "all":
            return captions
        elif isinstance(self._mode, int) and 0 <= self._mode < len(captions):
            index = self._mode
        elif (
            isinstance(self._mode, Iterable)
            and not isinstance(self._mode, str)
            and all(0 <= idx < len(captions) for idx in self._mode)
        ):
            local_index = int(torch.randint(0, len(self._mode), ()).item())
            index = self._mode[local_index]
        else:
            raise ValueError(
                f"Invalid argument mode={self._mode}. (expected 'random', 'all', int or Iterable of ints)"
            )

        if not self._keep_list:
            return captions[index]
        else:
            return [captions[index]]

    def extra_repr(self) -> str:
        return f"mode={self._mode}, keep_list={self._keep_list}"


class RandomSelect(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, captions: list[Any]) -> Any:
        index = random.randint(0, len(captions) - 1)
        return captions[index]
