#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os.path as osp

from typing import Optional

import torch

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback

from dcase2022task6a.nn.functional.misc import checksum


logger = logging.getLogger(__name__)


class ResumeCallback(Callback):
    def __init__(
        self,
        pl_resume_path: Optional[str],
        verbose: int = 1,
    ) -> None:
        super().__init__()
        self._pl_resume_path = pl_resume_path
        self._verbose = verbose

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._pl_resume_path is None:
            return None

        if not isinstance(self._pl_resume_path, str) or not osp.exists(
            self._pl_resume_path
        ):
            raise ValueError(
                f"Invalid resume checkpoint fpath {self._pl_resume_path=}. (path does not exists)"
            )

        if osp.isfile(self._pl_resume_path):
            ckpt_fpath = self._pl_resume_path
        elif osp.isdir(self._pl_resume_path):
            ckpt_fpath = osp.join(self._pl_resume_path, "checkpoints", "best.ckpt")
            if not osp.isfile(ckpt_fpath):
                raise FileNotFoundError(
                    f"Cannot find checkpoint in {self._pl_resume_path=} (expected in {{pl_resume}}/checkpoints/best.ckpt)."
                )
        else:
            raise ValueError(f'Invalid path type "{self._pl_resume_path=}".')

        if self._verbose:
            logger.info(f"Loading pl_module from checkpoint {ckpt_fpath=}.")

        pl_module.setup("fit")

        if self._verbose:
            logger.debug(
                f"pl_module csum before resume weights = {checksum(pl_module)}"
            )

        # Load best model before training
        checkpoint_data = torch.load(ckpt_fpath, map_location=pl_module.device)
        pl_module.load_state_dict(checkpoint_data["state_dict"], strict=False)

        if self._verbose:
            logger.debug(f"pl_module csum after resume weights = {checksum(pl_module)}")
