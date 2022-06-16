#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Microsoft COCO caption metric 'SPICE'.

    Code imported from : https://github.com/peteanderson80/coco-caption/blob/master/pycocoevalcap/spice/spice.py
    Authors : Peter Anderson
    Modified : Yes (typing_, names, imports and attributes)
"""

__author__ = "Peter Anderson"

import json
import logging
import math
import os
import os.path as osp
import subprocess

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Optional, Union

import numpy as np
import torch

from torch import nn

from dcase2022task6a.metrics.utils import format_to_coco, check_java_path


SPICE_FNAME = osp.join("spice", "spice-1.0.jar")
CACHE_DNAME = "cache"


class Spice(nn.Module):
    """Main Class to compute the SPICE metric."""

    def __init__(
        self,
        java_path: str = "java",
        ext_dpath: Optional[str] = None,
        tmp_dpath: str = "/tmp",
        n_threads: Optional[int] = None,
        java_max_memory: str = "8G",
        verbose: bool = False,
    ) -> None:
        if not check_java_path(java_path):
            raise ValueError(
                f"Cannot find java executable with {java_path=} for compute {self.__class__.__name__} metric score."
            )

        if ext_dpath is None:
            try:
                ext_dpath = str(Path(__file__).parent.joinpath("java"))
            except NameError as err:
                raise RuntimeError(f"Cannot get the SPICE root. ({err})")
        spice_fpath = osp.join(ext_dpath, SPICE_FNAME)

        if not osp.isfile(spice_fpath):
            raise FileNotFoundError(
                f"Cannot find JAR file '{SPICE_FNAME}' in directory '{ext_dpath}' for {self.__class__.__name__} metric."
            )

        os.makedirs(tmp_dpath, exist_ok=True)

        cache_dpath = osp.join(tmp_dpath, CACHE_DNAME)
        os.makedirs(cache_dpath, exist_ok=True)

        super().__init__()
        self._java_path = java_path
        self._spice_path = ext_dpath
        self._tmp_path = tmp_dpath
        self._n_threads = n_threads
        self._java_max_memory = java_max_memory
        self._verbose = verbose

    def forward(
        self,
        hypotheses: list[str],
        references: list[list[str]],
        return_dict: bool = False,
    ) -> Union[float, dict]:
        if len(hypotheses) != len(references):
            raise ValueError(
                f"Number of hypothesis and references are different (found {len(hypotheses)=} != {len(references)=})."
            )

        res, gts = format_to_coco(hypotheses, references)
        score, scores = self.compute_score(gts, res)
        score = score.item()
        scores = torch.from_numpy(scores)

        self._last_score = score

        if not return_dict:
            return score
        else:
            return {
                "score": score,
                "scores": scores,
            }

    def get_last_score(self) -> float:
        return self._last_score

    def compute_score(
        self,
        gts: dict[Any, list],
        res: dict[Any, list],
    ) -> tuple[np.ndarray, np.ndarray]:
        assert set(gts.keys()) == set(res.keys())
        imgIds = sorted(gts.keys())

        # Prepare temp input file for the SPICE scorer
        input_data = []
        for id_ in imgIds:
            hypo = res[id_]
            ref = gts[id_]

            # Sanity check.
            assert type(hypo) is list
            assert len(hypo) == 1
            assert type(ref) is list
            assert len(ref) >= 1

            input_data.append(
                {
                    "image_id": id_,
                    "test": hypo[0],
                    "refs": ref,
                }
            )

        in_file = NamedTemporaryFile(
            mode="w", delete=False, dir=self._tmp_path, suffix=".json", prefix="in_"
        )
        json.dump(input_data, in_file, indent=2)
        in_file.close()

        out_file = NamedTemporaryFile(
            mode="w", delete=False, dir=self._tmp_path, suffix=".json", prefix="out_"
        )
        out_file.close()

        if self._verbose:
            stdout = None
            stderr = None
        else:
            stdout = subprocess.DEVNULL
            stderr = subprocess.DEVNULL

        cache_path = osp.join(self._tmp_path, CACHE_DNAME)
        spice_fpath = osp.join(self._spice_path, SPICE_FNAME)
        spice_cmd = [
            self._java_path,
            "-jar",
            f"-Xmx{self._java_max_memory}",
            spice_fpath,
            in_file.name,
            "-cache",
            cache_path,
            "-out",
            out_file.name,
            "-subset",
        ]
        if self._n_threads is not None:
            spice_cmd += ["-threads", str(self._n_threads)]

        if self._verbose:
            logging.debug(f"Run SPICE java code with: {' '.join(spice_cmd)}")

        try:
            subprocess.check_call(
                spice_cmd,
                stdout=stdout,
                stderr=stderr,
            )
        except (subprocess.CalledProcessError, PermissionError) as err:
            logging.error(
                f"Invalid SPICE call. (full_command='{' '.join(spice_cmd)}', {err=})"
            )
            raise err

        if self._verbose:
            logging.debug("SPICE java code finished.")

        # Read and process results
        with open(out_file.name, "r") as data_file:
            results = json.load(data_file)
        os.remove(in_file.name)
        os.remove(out_file.name)

        imgId_to_scores = {}
        spice_scores = []
        for item in results:
            imgId_to_scores[item["image_id"]] = item["scores"]
            spice_scores.append(_float_convert(item["scores"]["All"]["f"]))
        spice_scores = np.array(spice_scores)
        average_score = np.mean(spice_scores)
        scores = []
        for image_id in imgIds:
            # Convert none to NaN before saving scores over subcategories
            score_set = {}
            for category, score_tuple in imgId_to_scores[image_id].items():
                score_set[category] = {
                    k: _float_convert(v) for k, v in score_tuple.items()
                }
            scores.append(score_set)
        return average_score, spice_scores

    def method(self) -> str:
        return "SPICE"


def _float_convert(obj: Any) -> float:
    try:
        return float(obj)
    except (ValueError, TypeError):
        return math.nan
