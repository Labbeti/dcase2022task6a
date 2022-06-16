#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Union

import torch

from torch import Tensor


def multihots_to_ints(multihots: Tensor) -> Union[list, Tensor]:
    if multihots.ndim == 0:
        return torch.empty((0,), dtype=torch.int, device=multihots.device)
    elif multihots.ndim == 1:
        arange = torch.arange(len(multihots), device=multihots.device)
        indexes = arange[multihots.cpu().bool()]
        return indexes
    else:
        return [multihots_to_ints(value) for value in multihots]


def ints_to_multihots(
    ints: Union[list[int], Tensor],
    n_classes: Optional[int],
    dtype: torch.dtype = torch.bool,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Returns multihots encoded version of ints."""
    if isinstance(ints, list) and not all(isinstance(v, int) for v in ints):
        raise TypeError(
            "Invalid argument type. (expected list[int] or IntTensor but found a list of non-integers objects)"
        )
    elif isinstance(ints, Tensor) and ints.is_floating_point():
        raise TypeError(
            "Invalid argument type. (expected list[int] or IntTensor but found floating point tensor)"
        )

    if isinstance(ints, list):
        ints = torch.as_tensor(ints, device=device)
    elif device is None:
        device = ints.device

    if isinstance(ints, Tensor):
        ints_shape = tuple(ints.shape)
    else:
        ints_shape = (len(ints),)

    if n_classes is None:
        n_classes = int(ints.max().item()) + 1

    multihots = torch.zeros(ints_shape + (n_classes,), device=device)
    multihots.scatter_(-1, ints.unsqueeze(dim=-1), 1.0)
    multihots = multihots.sum(dim=-2).clamp(max=1.0).to(dtype=dtype)
    return multihots
