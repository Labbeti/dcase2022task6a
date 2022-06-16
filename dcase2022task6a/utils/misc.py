#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import inspect
import os.path as osp
import pickle
import random
import subprocess
import zlib

from typing import Any, Callable, Iterable, MutableSequence, Optional, Sequence, Union

import hashlib
import numpy as np
import torch

from omegaconf import DictConfig
from torch import nn, Tensor


def get_none() -> None:
    # Returns None. Can be used for hydra instantiations.
    return None


def get_datetime() -> str:
    now = datetime.datetime.now()
    return now.strftime("%Y:%m:%d_%H:%M:%S")


def md5_int(x: bytes) -> int:
    x = hashlib.md5(x).digest()
    return int.from_bytes(x, "big", signed=False)


def reset_seed(seed: Optional[int]) -> None:
    """Reset the seed of following packages for reproductibility :
    - random
    - numpy
    - torch
    - torch.cuda

    Also set deterministic behaviour for cudnn backend.

    :param seed: The seed to set.
    """
    if seed is not None and not isinstance(seed, int):
        raise TypeError(
            f"Invalid argument type {type(seed)=}. (expected NoneType or int)"
        )

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False  # type: ignore
        torch.backends.cudnn.deterministic = True  # type: ignore


def get_current_git_hash(cwd: str = osp.dirname(__file__)) -> str:
    """
    Return the current git hash in the current directory.

    :returns: The git hash. If an error occurs, returns 'UNKNOWN'.
    """
    try:
        git_hash = subprocess.check_output("git describe --always".split(" "), cwd=cwd)
        git_hash = git_hash.decode("UTF-8").replace("\n", "")
        return git_hash
    except (subprocess.CalledProcessError, PermissionError):
        return "UNKNOWN"


def get_tags_version(cwd: str = osp.dirname(__file__)) -> str:
    """
    {LAST_TAG}-{NB_COMMIT_AFTER_LAST_TAG}-g{LAST_COMMIT_HASH}
    Example : v0.1.1-119-g40317c7

    :returns: The tag version with the git hash.
    """
    try:
        git_hash = subprocess.check_output("git describe --tags".split(" "), cwd=cwd)
        git_hash = git_hash.decode("UTF-8").replace("\n", "")
        return git_hash
    except (subprocess.CalledProcessError, PermissionError):
        return "UNKNOWN"


def split_indexes(
    indexes: MutableSequence[int],
    ratios: Sequence[float],
    shuffle: bool = False,
) -> list[list[int]]:

    if not (0.0 <= sum(ratios) <= 1.0 + 1e-15):
        raise ValueError(f"Ratio sum {sum(ratios)} must be in range [0, 1].")

    if shuffle:
        random.shuffle(indexes)

    prev_idx = 0
    split = []
    for ratio in ratios:
        next_idx = prev_idx + round(len(indexes) * ratio)
        split.append(indexes[prev_idx:next_idx])
        prev_idx = next_idx
    return split


def cache_feature(func: Callable) -> Callable:
    def decorator(*args, **kwargs):
        key = ",".join(map(str, args + tuple(kwargs.items())))

        if key not in decorator.cache:
            decorator.cache[key] = func(*args, **kwargs)

        return decorator.cache[key]

    decorator.cache = dict()
    decorator.func = func
    return decorator


def all_eq(it: Iterable, eq_func: Optional[Callable] = None) -> bool:
    try:
        it = iter(it)
        first = next(it)
        if eq_func is None:
            return all(first == elt for elt in it)
        else:
            return all(eq_func(first, elt) for elt in it)
    except StopIteration:
        return True


def list_dict_to_dict_list(
    lst: Sequence[dict[str, Any]],
    default_val: Any = None,
    error_on_missing_key: bool = False,
) -> dict[str, list[Any]]:
    """
    Convert a list of dicts to a dict of lists.

    Example
    ----------
    >>> lst = [{'a': 1, 'b': 2}, {'a': 4, 'b': 3, 'c': 5}]
    >>> output = list_dict_to_dict_list(lst, default_val=0)
    {'a': [1, 4], 'b': [2, 3], 'c': [0, 5]}
    """
    if len(lst) == 0:
        return {}

    if error_on_missing_key:
        keys = set(lst[0])
        for dic in lst:
            if keys != set(dic.keys()):
                raise ValueError(
                    f"Invalid dict keys for list_dict_to_dict_list. (found {keys} and {dic.keys()})"
                )

    keys = {}
    for dic in lst:
        keys = keys | dict.fromkeys(dic.keys())
    out = {
        key: [
            lst[i][key] if key in lst[i].keys() else default_val
            for i in range(len(lst))
        ]
        for key in keys
    }
    return out


def dict_list_to_list_dict(dic: dict[str, list[Any]]) -> list[dict[str, Any]]:
    assert all_eq(map(len, dic.values()))
    length = len(next(iter(dic.values())))
    return [{k: v[i] for k, v in dic.items()} for i in range(length)]


def flat_dict(dic: dict[str, Any], join_str: str = ".") -> dict:
    output = {}
    for k, v in dic.items():
        if isinstance(v, dict):
            v = flat_dict(v, join_str)
            for kv, vv in v.items():
                output[f"{k}{join_str}{kv}"] = vv
        else:
            output[k] = v
    return output


def any_checksum(
    x: Any,
    max_value: Optional[int] = None,
    out_type: Optional[Callable] = int,
    order: bool = False,
    csum_type: Union[str, Callable[[bytes], int]] = "md5",
    start_csum: int = 0,
) -> int:
    if csum_type == "adler32":
        csum_type = zlib.adler32
    elif csum_type == "md5":
        csum_type = md5_int
    elif isinstance(csum_type, Callable):
        pass
    else:
        raise ValueError(
            f"Invalid argument {csum_type=}. (expected 'adler32', 'md5' or a custom callable function)"
        )

    def any_checksum_rec(x: Any, accumulator: int) -> int:
        if x is None:
            return accumulator
        elif isinstance(x, bytes):
            return csum_type(x) + accumulator
        elif isinstance(x, str):
            return any_checksum_rec(x.encode(), accumulator)
        elif isinstance(x, (int, float, bool)):
            return any_checksum_rec(str(x), accumulator)
        elif isinstance(x, dict):
            return any_checksum_rec(list(x.items()), accumulator)
        elif isinstance(x, DictConfig):
            return any_checksum_rec(dict(x), accumulator)
        elif isinstance(x, type) or inspect.isfunction(x) or inspect.ismethod(x):
            return any_checksum_rec(x.__name__, accumulator)
        elif isinstance(x, Tensor):
            return any_checksum_rec(str(x.tolist()), accumulator)
        elif isinstance(x, nn.Module):
            return any_checksum_rec(x.named_parameters(), accumulator)
        elif isinstance(x, Iterable):
            return any_checksum_rec(
                sum(
                    any_checksum_rec(elt, accumulator + i if order else accumulator)
                    for i, elt in enumerate(x)
                ),
                accumulator,
            )
        else:
            try:
                dumped = pickle.dumps(x)
                return any_checksum_rec(dumped, accumulator)
            except TypeError:
                raise TypeError(
                    f"Unsupported type '{x.__class__.__name__}' for function any_checksum."
                )

    output = any_checksum_rec(x, start_csum)
    if max_value is not None:
        output = output % max_value
    if out_type is not None:
        output = out_type(output)
    return output


def filter_and_call(func: Callable, kwargs: dict[str, Any]) -> Any:
    varnames = func.__code__.co_varnames
    kwargs_filtered = {
        name: value for name, value in kwargs.items() if name in varnames
    }
    return func(**kwargs_filtered)


def prod_rec(x: Union[int, Iterable]) -> int:
    if isinstance(x, int):
        return x
    elif isinstance(x, Iterable):
        out = 1
        for xi in x:
            out *= prod_rec(xi)
        return out
    else:
        raise TypeError(f"Invalid argument type {type(x)=}.")


def flat_list_rec(nested_lst: list) -> tuple[Union[list, tuple], Union[list, tuple]]:
    if not isinstance(nested_lst, (list, tuple)):
        return (nested_lst,), ()
    else:
        flat_lst = []
        shapes = []
        for elt in nested_lst:
            subelt, subshapes = flat_list_rec(elt)
            flat_lst += subelt
            shapes.append(subshapes)

        if len(shapes) == 0:
            return [], (0,)
        elif all(subshapes == shapes[0] for subshapes in shapes):
            return flat_lst, (len(nested_lst),) + shapes[0]
        else:
            return flat_lst, shapes


def unflat_list_rec(flat_lst: list, shapes: Union[list, tuple]) -> list:
    if isinstance(shapes, tuple):
        if shapes == ():
            return flat_lst[0]
        else:
            array = np.array(flat_lst, dtype=object)
            array = array.reshape(*shapes)
            array = array.tolist()
            return array
    else:
        out = []
        idx = 0
        for shape_i in shapes:
            num_elements = prod_rec(shape_i)
            unflatten = unflat_list_rec(flat_lst[idx : idx + num_elements], shape_i)
            idx += num_elements
            out.append(unflatten)
        return out


def filter_kwargs(class_, kwargs: dict) -> dict:
    varnames = class_.__init__.__code__.co_varnames
    return {name: value for name, value in kwargs.items() if name in varnames}


def get_obj_fullname(obj: Any) -> str:
    class_ = obj.__class__
    module = class_.__module__
    if module == "builtins":
        return class_.__qualname__  # avoid outputs like 'builtins.str'
    return module + "." + class_.__qualname__
