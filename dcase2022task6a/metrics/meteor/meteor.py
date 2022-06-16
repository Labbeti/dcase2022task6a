#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help

# =================================================================
# This code was pulled from https://github.com/tylin/coco-caption
# and refactored for Python 3.
# =================================================================

import logging
import os.path as osp
import subprocess
import threading

from typing import Union

import torch

from torch import nn

from dcase2022task6a.metrics.utils import format_to_coco, check_java_path


logger = logging.getLogger(__name__)


METEOR_FNAME = osp.join("meteor", "meteor-1.5.jar")


class Meteor(nn.Module):
    def __init__(
        self,
        java_path: str = "java",
        ext_dpath: str = "ext",
        java_max_memory: str = "2G",
        verbose: int = 0,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        meteor_jar_fpath = osp.join(ext_dpath, METEOR_FNAME)
        meteor_command = [
            java_path,
            "-jar",
            f"-Xmx{java_max_memory}",
            meteor_jar_fpath,
            "-",
            "-",
            "-stdio",
            "-l",
            "en",
            "-norm",
        ]

        if not check_java_path(java_path):
            raise ValueError(
                f"Cannot find java executable with {java_path=} for compute {self.__class__.__name__} metric score."
            )
        if not osp.isfile(meteor_jar_fpath):
            raise FileNotFoundError(
                f"Cannot find JAR file '{METEOR_FNAME}' in directory '{ext_dpath}' for {self.__class__.__name__} metric."
            )
        if verbose >= 2:
            logger.debug(
                f"Start METEOR process with command '{' '.join(meteor_command)}'..."
            )

        super().__init__()
        self.verbose = verbose
        self.dtype = dtype

        self._meteor_command = meteor_command
        self._meteor_p = subprocess.Popen(
            self._meteor_command,
            cwd=osp.dirname(osp.abspath(__file__)),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Used to guarantee thread safety
        self._lock = threading.Lock()

    def forward(
        self,
        hypotheses: list[str],
        references: list[list[str]],
        return_dict: bool = False,
    ) -> Union[float, dict]:
        res, gts = format_to_coco(hypotheses, references)
        score, scores = self.compute_score(gts, res)
        score = score
        scores = torch.as_tensor(scores, dtype=self.dtype)

        if not return_dict:
            return score
        else:
            return {
                "score": score,
                "scores": scores,
            }

    def compute_score(self, gts: dict, res: dict) -> tuple[float, list[float]]:
        assert gts.keys() == res.keys()
        imgIds = gts.keys()
        scores = []

        eval_line = "EVAL"
        self._lock.acquire()
        for i in imgIds:
            assert len(res[i]) == 1
            stat = self._stat(res[i][0], gts[i])
            eval_line += " ||| {}".format(stat)

        assert self._meteor_p.stdin is not None
        if self.verbose >= 2:
            logger.debug(f"Write line {eval_line=}.")
        self._meteor_p.stdin.write("{}\n".format(eval_line).encode())
        self._meteor_p.stdin.flush()
        assert self._meteor_p.stdout is not None

        for i in range(0, len(imgIds)):
            # TODO : find fix when line is empty ?
            try:
                scores.append(float(self._meteor_p.stdout.readline().strip()))
            except ValueError as err:
                logger.error(f"{i=}")
                logger.error(f"{gts[list(imgIds)[i]]=}")
                logger.error(f"{res[list(imgIds)[i]]=}")
                logger.error(f"{self._meteor_p.pid=}")
                raise err

        score = float(self._meteor_p.stdout.readline().strip())
        self._lock.release()

        return score, scores

    def method(self) -> str:
        return "METEOR"

    def _stat(self, hypothesis_str: str, reference_list: list[str]) -> str:
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace("|||", "").replace("  ", " ")
        score_line = " ||| ".join(
            ("SCORE", " ||| ".join(reference_list), hypothesis_str)
        )
        assert self._meteor_p.stdin is not None
        self._meteor_p.stdin.write("{}\n".format(score_line).encode())
        self._meteor_p.stdin.flush()
        assert self._meteor_p.stdout is not None
        return self._meteor_p.stdout.readline().decode().strip()

    def _score(self, hypothesis_str: str, reference_list: list[str]) -> float:
        self._lock.acquire()
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace("|||", "").replace("  ", " ")
        score_line = " ||| ".join(
            ("SCORE", " ||| ".join(reference_list), hypothesis_str)
        )
        assert self._meteor_p.stdin is not None
        self._meteor_p.stdin.write("{}\n".format(score_line).encode())
        assert self._meteor_p.stdout is not None
        stats = self._meteor_p.stdout.readline().strip()
        eval_line = "EVAL ||| {}".format(stats)
        # EVAL ||| stats
        self._meteor_p.stdin.write("{}\n".format(eval_line).encode())
        score = float(self._meteor_p.stdout.readline().strip())
        # bug fix: there are two values returned by the jar file, one average, and one all, so do it twice
        # thanks for Andrej for pointing this out
        score = float(self._meteor_p.stdout.readline().strip())
        self._lock.release()
        return score

    def _kill_process(self) -> None:
        if hasattr(self, "_lock") and self._lock is not None:
            self._lock.acquire()
            assert self._meteor_p.stdin is not None
            self._meteor_p.stdin.close()
            self._meteor_p.kill()
            self._meteor_p.wait()
            self._lock.release()

    def __exit__(self) -> None:
        self._kill_process()
