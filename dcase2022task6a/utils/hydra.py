#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path as osp
import pickle

from pathlib import Path
from typing import Any, Union

from omegaconf import DictConfig, OmegaConf


def include_keys_func(prefix: str, _root_: DictConfig) -> list[str]:
    included = OmegaConf.select(_root_, key=prefix).keys()
    overrides = _root_.hydra.overrides.task
    overrides_keys = [
        key_value.split("=")[0].removeprefix("+") for key_value in overrides
    ]
    excluded = [value for value in overrides_keys if value not in included]
    return excluded


def get_none() -> None:
    # Returns None. Can be used for hydra instantiations.
    return None


def get_pickle(
    fpath: Union[str, Path],
) -> Any:
    if not isinstance(fpath, (str, Path)):
        raise TypeError(f"Invalid transform with pickle {fpath=}. (not a str or Path)")
    if not osp.isfile(fpath):
        raise FileNotFoundError(f"Invalid transform with pickle {fpath=}. (not a file)")

    with open(fpath, "rb") as file:
        data = pickle.load(file)
    return data
