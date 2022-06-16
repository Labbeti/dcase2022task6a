#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os.path as osp


# source : https://github.com/emrcak/dcase-2020-baseline/tree/sed_caps/data/cleanup_files
FUNCTION_WORDS_LINKS = {
    "articles": "https://raw.githubusercontent.com/emrcak/dcase-2020-baseline/sed_caps/data/cleanup_files/articles.dat",
    "auxiliary_verbs": "https://raw.githubusercontent.com/emrcak/dcase-2020-baseline/sed_caps/data/cleanup_files/auxiliary_verbs.dat",
    "conjunctions": "https://raw.githubusercontent.com/emrcak/dcase-2020-baseline/sed_caps/data/cleanup_files/conjunctions.dat",
    "prepositions": "https://raw.githubusercontent.com/emrcak/dcase-2020-baseline/sed_caps/data/cleanup_files/prepositions.dat",
}


def load_function_words(dataroot: str, verbose: bool = False) -> dict[str, None]:
    if dataroot is None or not osp.isdir(dataroot):
        raise RuntimeError(f"Invalid directory {dataroot=}.")
    func_words_dpath = osp.join(dataroot, "meta", "function_words")

    func_words = []
    for name in FUNCTION_WORDS_LINKS.keys():
        fname = f"{name}.dat"
        fpath = osp.join(func_words_dpath, fname)
        if osp.isfile(fpath):
            if verbose:
                logging.info(f"Loading {name} from '{fpath=}'...")

            with open(fpath, "r") as file:
                lines = file.readlines()
                lines = [line.replace("\n", "") for line in lines]
                func_words += lines

    func_words = dict.fromkeys(func_words)
    return func_words


def load_words_alpha(dataroot: str) -> dict[str, None]:
    if dataroot is None or not osp.isdir(dataroot):
        raise RuntimeError(f"Invalid directory {dataroot=}.")
    words_list_fpath = osp.join(dataroot, "meta", "words_alpha.txt")

    with open(words_list_fpath, "r") as file:
        words = file.readlines()
    words = dict.fromkeys(words)
    return words
