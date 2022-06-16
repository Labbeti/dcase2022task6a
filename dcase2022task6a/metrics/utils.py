#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import subprocess

from pathlib import Path
from subprocess import CalledProcessError
from typing import Hashable, Optional, Sequence, Union


def format_to_coco(
    hypotheses: list[str],
    references: list[list[str]],
    ids: Optional[Sequence[Hashable]] = None,
) -> tuple[dict[Hashable, list[str]], dict[Hashable, list[str]]]:
    """Format hypotheses and references to COCO metrics input dict (res, gts)."""
    if len(hypotheses) != len(references):
        raise ValueError(
            f"Invalid number of hypotheses={len(hypotheses)} with number of references={len(references)}."
        )
    if ids is None:
        ids = list(range(len(hypotheses)))
    res = {id_: [hyp] for id_, hyp in zip(ids, hypotheses)}
    gts = {id_: list(refs) for id_, refs in zip(ids, references)}
    return res, gts


def check_java_path(java_path: Union[str, Path]) -> bool:
    if not isinstance(java_path, (str, Path)):
        return False

    try:
        exitcode = subprocess.check_call(
            [java_path, "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (CalledProcessError, PermissionError, FileNotFoundError):
        exitcode = 1
    return exitcode == 0


def check_word_level(
    hypotheses: Sequence[Sequence[str]],
    references: Sequence[Sequence[list[str]]],
    *args,
    **kwargs,
) -> bool:
    valid = True
    if not isinstance(hypotheses, Sequence):
        logging.error(f"Invalid word_level: {isinstance(hypotheses, Sequence)=}")
        valid = False

    if not isinstance(references, Sequence):
        logging.error(f"Invalid word_level: {isinstance(references, Sequence)=}")
        valid = False

    if not len(hypotheses) == len(references):
        logging.error(f"Invalid word_level: {len(hypotheses) == len(references)=}")
        valid = False

    if not all(isinstance(hyp, Sequence) for hyp in hypotheses):
        logging.error(
            f"Invalid word_level: {all(isinstance(hyp, Sequence) for hyp in hypotheses)=}"
        )
        valid = False

    if not all(isinstance(refs, Sequence) for refs in references):
        logging.error(
            f"Invalid word_level: {all(isinstance(refs, Sequence) for refs in references)=}"
        )
        valid = False

    if not all(len(refs) > 0 for refs in references):
        logging.error(
            f"Invalid word_level: {all(len(refs) > 0 for refs in references)=}"
        )
        valid = False

    if not all(isinstance(ref, Sequence) for refs in references for ref in refs):
        logging.error(
            f"Invalid word_level: {all(isinstance(ref, Sequence) for refs in references for ref in refs)=}"
        )
        valid = False

    if not all(len(ref) > 0 for refs in references for ref in refs):
        logging.error(
            f"Invalid word_level: {all(len(ref) > 0 for refs in references for ref in refs)=}"
        )
        valid = False

    if not all(isinstance(token, str) for hyp in hypotheses for token in hyp):
        logging.error(
            f"Invalid word_level: {all(isinstance(token, str) for hyp in hypotheses for token in hyp)=}"
        )
        valid = False

    if not all(
        isinstance(token, str) for refs in references for ref in refs for token in ref
    ):
        logging.error(
            f"Invalid word_level: {all(isinstance(token, str) for refs in references for ref in refs for token in ref)=}"
        )
        valid = False

    return valid


def check_sentence_level(
    hypotheses: Sequence[str],
    references: Sequence[list[str]],
    *args,
    **kwargs,
) -> bool:
    valid = True
    if not isinstance(hypotheses, Sequence):
        logging.error(f"Invalid sent_level: {isinstance(hypotheses, Sequence)=}")
        valid = False

    if not isinstance(references, Sequence):
        logging.error(f"Invalid sent_level: {isinstance(references, Sequence)=}")
        valid = False

    if not len(hypotheses) == len(references):
        logging.error(f"Invalid sent_level: {len(hypotheses) == len(references)=}")
        valid = False

    if not all(isinstance(hyp, str) for hyp in hypotheses):
        logging.error(
            f"Invalid sent_level: {all(isinstance(hyp, str) for hyp in hypotheses)=}"
        )
        logging.error(f"\texample: {hypotheses[0]=}")
        valid = False

    if not all(isinstance(refs, list) for refs in references):
        logging.error(
            f"Invalid sent_level: {all(isinstance(refs, list) for refs in references)=}"
        )
        valid = False

    if not all(len(refs) > 0 for refs in references):
        logging.error(
            f"Invalid sent_level: {all(len(refs) > 0 for refs in references)=}"
        )
        valid = False

    if not all(isinstance(ref, str) for refs in references for ref in refs):
        logging.error(
            f"Invalid sent_level: {all(isinstance(ref, str) for refs in references for ref in refs)=}"
        )
        logging.error(f"\texample: {references[0][0]=}")
        valid = False

    return valid
