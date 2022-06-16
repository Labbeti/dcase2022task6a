#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import logging
import os
import os.path as osp

from typing import Any, Optional

import yaml

from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.core.saving import save_hparams_to_yaml
from torch import Tensor

from dcase2022task6a.callbacks.time import TimeTrackerCallback
from dcase2022task6a.callbacks.utils import get_pl_loggers
from dcase2022task6a.nn.functional.misc import count_params, checksum
from dcase2022task6a.tokenization import AACTokenizer
from dcase2022task6a.utils.custom_logger import CustomTensorboardLogger
from dcase2022task6a.utils.misc import get_current_git_hash
from dcase2022task6a.version import get_packages_versions


logger = logging.getLogger(__name__)


class StatsSaver(Callback):
    """Callback for saving some stats about the training in the logger."""

    def __init__(
        self,
        logdir: Optional[str],
        tokenizers: Optional[dict[str, AACTokenizer]] = None,
        on_end: str = "test",
        close_logger_on_end: bool = True,
        verbose: int = 1,
    ) -> None:
        if on_end not in ("fit", "test", "none"):
            raise ValueError(f"Invalid argument {on_end=}.")
        if tokenizers is None:
            tokenizers = {}
        tokenizers = {
            name: tokenizer
            for name, tokenizer in tokenizers.items()
            if tokenizer is not None
        }

        super().__init__()
        self._logdir = logdir
        self._tokenizers = tokenizers
        self._on_end = on_end
        self._close_logger_on_end = close_logger_on_end
        self._verbose = verbose

        self._time_tracker = TimeTrackerCallback()
        self._git_hash = get_current_git_hash()
        self._active = True
        self._start_csum = 0.0
        self._end_csum = 0.0

    def set_active_save(self, active: bool) -> None:
        self._active = active

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._time_tracker.on_fit_start(trainer, pl_module)
        self._start_csum = checksum(pl_module)
        self._end_csum = self._start_csum

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._time_tracker.on_fit_end(trainer, pl_module)
        self._end_csum = checksum(pl_module)

        if self._on_end == "fit" and self._active:
            self._save_stats(trainer, pl_module)

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._time_tracker.on_test_start(trainer, pl_module)

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._time_tracker.on_test_end(trainer, pl_module)
        if self._on_end == "test" and self._active:
            self._save_stats(trainer, pl_module)

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self._time_tracker.on_train_epoch_start(trainer, pl_module)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._time_tracker.on_train_epoch_end(trainer, pl_module)

    def _save_stats(self, trainer: Any, pl_module: LightningModule) -> None:
        datamodule = trainer.datamodule

        versions = get_packages_versions()
        versions = {f"version_{name}": version for name, version in versions.items()}

        params = {
            "fit_duration": self._time_tracker.get_fit_duration_formatted(),
            "test_duration": self._time_tracker.get_test_duration_formatted(),
            "git_hash": self._git_hash,
        }
        params |= {
            key.lower(): value
            for key, value in os.environ.items()
            if key.startswith("SLURM_")
        }
        params |= versions

        other_metrics = {
            "total_params": count_params(pl_module, only_trainable=False),
            "train_params": count_params(pl_module, only_trainable=True),
            "start_csum": self._start_csum,
            "end_csum": self._end_csum,
            "fit_duration_hours": self._time_tracker.get_fit_duration_in_hours(),
            "test_duration_hours": self._time_tracker.get_test_duration_in_hours(),
            "epoch_mean_duration_min": self._time_tracker.get_epoch_mean_duration_in_min(),
        }

        checkpoint = trainer.checkpoint_callback
        if checkpoint is not None and hasattr(
            checkpoint, "get_best_monitor_candidates"
        ):
            best_monitor_candidates = checkpoint.get_best_monitor_candidates()

            def convert(value) -> Any:
                if isinstance(value, Tensor):
                    if value.nelement() == 1:
                        return value.item()
                    else:
                        return value.tolist()
                else:
                    return value

            best_monitor_candidates = {
                f"best_{name}": convert(value)
                for name, value in best_monitor_candidates.items()
            }
            other_metrics |= best_monitor_candidates

        if isinstance(self._logdir, str) and osp.isdir(self._logdir):
            for name, tokenizer in self._tokenizers.items():
                if tokenizer is None:
                    continue

                # Save tokenizer to pickle file
                tokenizer_fname = f"{name}.pickle"
                tokenizer_fpath = osp.join(self._logdir, tokenizer_fname)
                tokenizer.save(tokenizer_fpath)

                if hasattr(tokenizer, "get_hparams") or hasattr(tokenizer, "hparams"):
                    hparams = (
                        tokenizer.get_hparams()
                        if hasattr(tokenizer, "get_hparams")
                        else tokenizer.hparams
                    )
                    # Save tokenizer hparams to yaml file
                    hparams_fname = f"hparams_{name}.yaml"
                    hparams_fpath = osp.join(self._logdir, hparams_fname)
                    with open(hparams_fpath, "w") as file:
                        yaml.dump(hparams, file)

                if isinstance(tokenizer, AACTokenizer) and tokenizer.is_fit():
                    # Save vocabulary to csv file
                    vocab_fname = f"vocabulary_{name}.csv"
                    vocab_fpath = osp.join(self._logdir, vocab_fname)

                    fieldnames = ("token", "occurrence", "index")
                    data = [
                        {
                            "token": token,
                            "occurrence": occurrence,
                            "index": tokenizer.stoi(token),
                        }
                        for token, occurrence in tokenizer.get_vocab().items()
                    ]

                    with open(vocab_fpath, "w") as file:
                        writer = csv.DictWriter(file, fieldnames)
                        writer.writeheader()
                        writer.writerows(data)

                    other_metrics[f"{name}_vocab_size"] = tokenizer.get_vocab_size()
                    other_metrics[
                        f"{name}_min_sentence_size"
                    ] = tokenizer.get_min_sentence_size()
                    other_metrics[
                        f"{name}_max_sentence_size"
                    ] = tokenizer.get_max_sentence_size()

            if pl_module is not None:
                save_hparams_to_yaml(
                    osp.join(self._logdir, "hparams_pl_module.yaml"),
                    pl_module.hparams_initial,
                )
            if datamodule is not None:
                save_hparams_to_yaml(
                    osp.join(self._logdir, "hparams_datamodule.yaml"),
                    datamodule.hparams_initial,
                )

        other_metrics = {
            f"other/{name}": value for name, value in other_metrics.items()
        }

        if self._verbose >= 2:
            logger.debug(
                f"Adding {len(params)} params :\n{yaml.dump(params, sort_keys=False)}"
            )
        if self._verbose >= 1:
            logger.info(
                f"Adding {len(other_metrics)} metrics :\n{yaml.dump(other_metrics, sort_keys=False)}"
            )

        pl_loggers = get_pl_loggers(pl_module)
        for pl_logger in pl_loggers:
            if isinstance(pl_logger, CustomTensorboardLogger):
                pl_logger.log_hyperparams(params=params, metrics=other_metrics)

                if self._close_logger_on_end:
                    pl_logger.save_and_close()
            else:
                pl_logger.log_hyperparams(params)
                pl_logger.log_metrics(other_metrics)
