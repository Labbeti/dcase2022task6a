#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import LoggerCollection


def get_pl_loggers(pl_module: LightningModule) -> list:
    pl_logger = pl_module.logger
    if pl_logger is None:
        pl_loggers = []
    elif isinstance(pl_logger, LoggerCollection):
        pl_loggers = list(pl_logger._logger_iterable)
    else:
        pl_loggers = [pl_logger]
    return pl_loggers
