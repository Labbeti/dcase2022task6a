#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Iterator

from torch import Tensor
from torch.optim import Optimizer, Adam, AdamW, SGD


def get_optimizer(
    optim_name: str,
    parameters: Iterator[Tensor],
    **kwargs,
) -> Optimizer:
    classes = (
        Adam,
        AdamW,
        SGD,
    )
    optimizer = None
    for class_ in classes:
        if optim_name.lower() == class_.__name__.lower():
            optimizer = class_(parameters, **_filter_kwargs(class_, kwargs))
            break

    if optimizer is not None:
        return optimizer
    else:
        try:
            optimizer = get_lib_package(optim_name, parameters, **kwargs)
            return optimizer
        except (ImportError, RuntimeError):
            raise RuntimeError(f'Unknown optimizer "{optim_name}".')


def get_lib_package(
    optim_name: str,
    parameters: Iterator[Tensor],
    **kwargs,
) -> Optimizer:
    import torch_optimizer as to

    classes = (
        to.AdaBelief,
        to.AdaMod,
        to.Adahessian,
        to.AdamP,
        to.Apollo,
        to.DiffGrad,
        to.Lamb,
        to.QHAdam,
        to.RAdam,
        to.Ranger,
        to.RangerVA,
        to.SWATS,
        to.Shampoo,
        to.Yogi,
    )

    optimizer = None
    for class_ in classes:
        if optim_name.lower() == class_.__name__.lower():
            optimizer = class_(parameters, **_filter_kwargs(class_, kwargs))
            break

    if optimizer is not None:
        return optimizer
    else:
        raise RuntimeError(f'Unknown optimizer "{optim_name}".')


def _filter_kwargs(class_, kwargs: dict) -> dict:
    varnames = class_.__init__.__code__.co_varnames
    return {name: value for name, value in kwargs.items() if name in varnames}
