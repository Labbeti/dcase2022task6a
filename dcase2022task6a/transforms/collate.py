#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from typing import Any, Sequence, Union

import torch

from torch import nn, Tensor

from dcase2022task6a.nn.functional.misc import pad_sequence_rec
from dcase2022task6a.utils.misc import all_eq


class CollateDict(nn.Module):
    """Collate list of dict into a dict of list WITHOUT auto-padding."""

    def forward(self, items_lst: list[dict[str, Any]]) -> dict[str, list[Any]]:
        keys = items_lst[0].keys()
        for i in range(1, len(items_lst)):
            keys = [key for key in keys if key in items_lst[i].keys()]
        return {key: [item[key] for item in items_lst] for key in keys}


class PadCollateDict(nn.Module):
    """Collate list of dict into a dict of list WITH auto-padding."""

    def __init__(
        self,
        pad_values: dict[str, Union[int, float]],
        skip_unknown_key: bool = True,
        exclude_keys: Sequence[str] = (),
    ) -> None:
        super().__init__()
        self._pad_values = pad_values
        self._skip_unknown_key = skip_unknown_key
        self._exclude_keys = exclude_keys

    def forward(self, items_lst: list[dict[str, Any]]) -> dict[str, Any]:
        # Intersection of keys and keep the same order
        keys = [key for key in items_lst[0].keys() if key not in self._exclude_keys]
        for item in items_lst[1:]:
            keys = [key for key in keys if key in item.keys()]

        batch = {}
        for key in keys:
            # Note : cannot use batch.items() because batch is modified during this loop
            items = [item[key] for item in items_lst]

            if all(isinstance(item, (int, float, str)) for item in items):
                pass

            elif all(isinstance(item, Tensor) for item in items):
                shapes = [item.shape for item in items]
                if not all_eq(map(len, shapes)):
                    logging.error(
                        f"Cannot collate list of tensors with a different number of dims. ({shapes=})"
                    )
                    continue

                shapes = torch.as_tensor(shapes)
                shape_key = f"{key}_shape"
                if not key.endswith("_shape") and shape_key not in self._exclude_keys:
                    if shape_key not in batch.keys():
                        batch[shape_key] = shapes
                    else:
                        if not batch[shape_key].eq(shapes).all():
                            logging.error(f"Fail sanity check {key=} with {shape_key=}")
                            continue

                if all(shape.eq(shapes[0]).all() for shape in shapes):
                    items = torch.stack(items, dim=0)
                elif key in self._pad_values.keys():
                    pad_value = self._pad_values[key]
                    items = pad_sequence_rec(items, pad_value=pad_value)
                elif self._skip_unknown_key:
                    pass
                else:
                    logging.error(
                        f"Invalid batch for {key=} in {self.__class__.__name__}."
                    )
                    continue

            elif all(isinstance(item, (Tensor, list, tuple)) for item in items):
                try:
                    scalar_type = detect_scalar_type(items)
                except RuntimeError:
                    logging.error(
                        f"Cannot detect scalar type in batch. ({key=}, {items=})"
                    )
                    continue

                if scalar_type in (Tensor, list, tuple, int, float):
                    try:
                        shapes = detect_shape(items)
                    except RuntimeError:
                        logging.error(
                            f"Cannot detect shape in batch. ({key=}, {items=}, {scalar_type=})"
                        )
                        continue
                    shape_key = f"{key}_shape"

                    if (
                        not key.endswith("_shape")
                        and shape_key not in self._exclude_keys
                    ):
                        if shape_key not in batch.keys():
                            batch[shape_key] = shapes
                        else:
                            if not batch[shape_key].eq(shapes).all():
                                logging.error(
                                    f"Fail sanity check {key=} with {shape_key=}"
                                )
                                continue

                    if scalar_type in (int, float) and len(torch.unique(shapes)) == 1:
                        items = torch.as_tensor(items)
                    elif key in self._pad_values.keys():
                        pad_value = self._pad_values[key]
                        items = pad_sequence_rec(items, pad_value=pad_value)
                    elif self._skip_unknown_key:
                        pass
                    else:
                        logging.error(
                            f"Invalid batch for {key=} in {self.__class__.__name__}."
                        )
                        continue
                elif scalar_type in (list, tuple, str):
                    pass
                else:
                    logging.error(f"Invalid item type {scalar_type=} with {key=}.")
                    continue

            else:
                logging.error(
                    f"Invalid item type {type(items[0])} with {key=}. (types={tuple(map(type, items))})"
                )
                continue

            batch[key] = items

        return batch


def detect_scalar_type(item: Any) -> type:
    types = set()
    queue = [item]
    while len(queue) > 0:
        item = queue.pop()
        if isinstance(item, (list, tuple)) and len(item) > 0:
            queue += item
        else:
            types.add(type(item))

    if len(types) == 1:
        return list(types)[0]
    else:
        raise RuntimeError(f"Multiple types detected: {types=}.")


def detect_shape(item: Any) -> Tensor:
    if isinstance(item, (int, float, str)):
        return torch.as_tensor((), dtype=torch.long)
    elif isinstance(item, Tensor) and item.ndim in (0, 1):
        return torch.as_tensor(item.shape, dtype=torch.long)
    elif isinstance(item, (Tensor, list, tuple)):
        if len(item) == 0 or isinstance(item[0], (int, float, str)):
            return torch.as_tensor((len(item),), dtype=torch.long)
        else:
            subshapes = [detect_shape(subitem) for subitem in item]
            subdims = list(map(len, subshapes))
            if not all_eq(subdims):
                logging.error(f"Function detech_shape: found {subshapes=}")
                raise RuntimeError(
                    f"Invalid number of dims with {subdims=} in function 'detect_shape'."
                )
            return torch.stack([torch.as_tensor(subshape) for subshape in subshapes])
    else:
        raise RuntimeError(f"Unknown subtype {item.__class__.__name__}.")
