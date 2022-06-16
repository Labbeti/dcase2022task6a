#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import re
import os
import os.path as osp

from typing import Any, Optional, Sequence

import yaml

from dcase2022task6a.tokenization import AACTokenizer


def get_hdf_fpaths(
    dataname: str,
    subsets: Sequence[str],
    hdf_root: str,
    hdf_suffix: Optional[str],
    hdf_dname: str = "HDF",
) -> dict[str, str]:
    """Returns the dictionary of HDF datasets filepaths for each subset :
    ```
    {
        {subset}: {hdf_root}/{hdf_dname}/{dataname}_{subset}_{hdf_suffix}.hdf
        {subset}: ...
    }
    ```
    If hdf_suffix is None, returns an empty dict.
    """
    if hdf_suffix is None:
        return {}

    dataname = dataname.lower()
    subsets = list(map(str.lower, subsets))
    pattern = re.compile(
        r"(?P<dataname>[a-z]+)_(?P<subset>[a-z]+)_(?P<hdf_suffix>.+)\.hdf"
    )

    hdf_fpaths = {}

    for subset in subsets:
        hdf_fname = f"{dataname}_{subset}_{hdf_suffix}.hdf"
        hdf_fpath = osp.join(hdf_root, hdf_dname, hdf_fname)

        if not osp.isfile(hdf_fpath):
            names = os.listdir(osp.join(hdf_root, hdf_dname))
            matches = [re.match(pattern, name) for name in names]
            availables_hdf_suffix = [
                match["hdf_suffix"]
                for match in matches
                if match is not None
                and match["dataname"] == dataname
                and match["subset"] == subset
            ]

            raise FileNotFoundError(
                f"Cannot find HDF file '{hdf_fpath}' with {hdf_suffix=}.\n"
                f"Maybe run dcase2022task6a.prepare before and use another hdf_suffix for {dataname}.\n"
                f"Available hdf_suffix for '{dataname}_{subset}' are:\n{yaml.dump(availables_hdf_suffix, sort_keys=False)}"
            )
        hdf_fpaths[subset] = hdf_fpath

    return hdf_fpaths


def filter_vocab_and_capsize(dset: Any, src_tokenizer: AACTokenizer) -> Sequence[int]:
    def pass_filter(idx: int) -> bool:
        captions_encoded = src_tokenizer.encode_batch(
            dset.get_raw("captions", idx),
            out_type=str,
            add_sos_eos=False,
            padding=False,
        )
        return all(
            src_tokenizer.get_min_sentence_size()
            <= len(caption)
            <= src_tokenizer.get_max_sentence_size()
            and all(token in src_tokenizer.get_vocab() for token in caption)
            for caption in captions_encoded
        )

    indexes = [idx for idx in range(len(dset)) if pass_filter(idx)]
    return indexes


def filter_capsize(
    dset: Any,
    src_tokenizer: AACTokenizer,
    min_cap_size: int,
    max_cap_size: int,
) -> Sequence[int]:
    def pass_filter(idx: int) -> bool:
        captions_encoded = src_tokenizer.encode_batch(dset.get_raw("captions", idx))
        return all(
            min_cap_size <= len(caption) <= max_cap_size for caption in captions_encoded
        )

    indexes = [idx for idx in range(len(dset)) if pass_filter(idx)]
    return indexes


def filter_en_words(
    dset: Any,
    src_tokenizer: AACTokenizer,
    meta_dpath: str,
) -> Sequence[int]:
    words_fname = "words_alpha.txt"
    words_fpath = osp.join(meta_dpath, words_fname)

    with open(words_fpath, "r") as file:
        lines = file.readlines()
    lines = map(lambda w: w.replace("\n", "").lower(), lines)
    en_vocabulary = dict.fromkeys(lines)

    def pass_filter(idx: int) -> bool:
        captions_encoded = src_tokenizer.encode_batch(
            dset.get_raw("captions", idx),
            out_type=str,
            add_sos_eos=False,
            padding=False,
        )
        return all(
            token in en_vocabulary for caption in captions_encoded for token in caption
        )

    indexes = [idx for idx in range(len(dset)) if pass_filter(idx)]
    return indexes


def split_indexes(
    indexes: Sequence[int],
    ratios: Sequence[float],
) -> list[Sequence[int]]:
    assert 0 <= sum(ratios) <= 1.0 + 1e-20, f"Found {sum(ratios)=} not in [0, 1]."
    ratio_cumsum = 0.0
    outs = []
    for ratio in ratios:
        start = math.floor(ratio_cumsum * len(indexes))
        end = math.floor((ratio_cumsum + ratio) * len(indexes))
        sub_indexes = indexes[start:end]
        outs.append(sub_indexes)
        ratio_cumsum += ratio
    return outs
