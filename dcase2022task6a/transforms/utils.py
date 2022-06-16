#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Optional

from torch import nn


class TransformDict(nn.Module):
    def __init__(
        self,
        transforms: Optional[dict[str, Optional[nn.Module]]] = None,
        **kwargs: Optional[nn.Module]
    ) -> None:
        if transforms is None:
            transforms = {}
        transforms |= kwargs
        super().__init__()
        self._transforms = transforms

        for name, transform in self._transforms.items():
            if transform is not None:
                self.add_module(name, transform)

    def forward(self, dic: dict[str, Any]) -> dict[str, Any]:
        out = {}
        for key, value in dic.items():
            transform = self._transforms.get(key, None)
            if transform is not None:
                value = transform(value)
            out[key] = value
        return dic
