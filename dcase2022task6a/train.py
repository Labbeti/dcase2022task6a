#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

os.environ["TRANSFORMERS_OFFLINE"] = "TRUE"

import logging
import os.path as osp
import sys

from typing import Sequence, Union

import hydra
import torch

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    ModelSummary,
)

from dcase2022task6a.callbacks.log import LogGCCallback, LogLRCallback, LogGradNorm, LogRngState
from dcase2022task6a.callbacks.resume import ResumeCallback
from dcase2022task6a.callbacks.stats_saver import StatsSaver
from dcase2022task6a.tokenization.aac_tokenizer import AACTokenizer
from dcase2022task6a.utils.handler import CustomFileHandler
from dcase2022task6a.utils.hydra import include_keys_func
from dcase2022task6a.utils.misc import reset_seed


logger = logging.getLogger(__name__)


@hydra.main(config_path=osp.join("..", "conf"), config_name="train")
def main_train(cfg: DictConfig) -> Union[None, float, Sequence[float]]:
    """Train a model on data."""
    reset_seed(cfg.seed)
    aac_logger = logging.getLogger("dcase2022task6a")
    if cfg.verbose <= 0:
        aac_logger.setLevel(logging.WARNING)
    elif cfg.debug:
        aac_logger.setLevel(logging.DEBUG)

    if cfg.verbose >= 1 and (cfg.tag != "NOTAG" or cfg.debug):
        logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
        logger.info(f"Datetime: {cfg.datetime}")

    if cfg.path.torch_hub is not None:
        torch.hub.set_dir(cfg.path.torch_hub)

    # Redicrect pytorch lightning outputs to a file.
    logdir = osp.join(cfg.log.save_dir, cfg.log.name, cfg.log.version)
    if osp.isdir(logdir):
        fpath_pl_outputs = osp.join(logdir, "logs", "lightning_outputs.log")
        pl_logger = logging.getLogger("pytorch_lightning")
        handler = CustomFileHandler(fpath_pl_outputs)
        pl_logger.addHandler(handler)

    if cfg.verbose >= 1:
        logger.info(f"Logdir: {logdir}")

    audio_trans_cfgs = {
        "train_audio_transform": cfg.audio_trans.train,
        "val_audio_transform": cfg.audio_trans.val,
        "test_audio_transform": cfg.audio_trans.test,
    }
    audio_transforms = {
        name: hydra.utils.instantiate(trans_cfg)
        for name, trans_cfg in audio_trans_cfgs.items()
    }

    fit_tokenizers = {
        "fit_tokenizer": hydra.utils.instantiate(cfg.fit_token),
        "fit_tokenizer_2": hydra.utils.instantiate(cfg.fit_token_2),
    }
    fit_tokenizers = {
        name: tokenizer
        for name, tokenizer in fit_tokenizers.items()
        if tokenizer is not None
    }

    test_tokenizer = hydra.utils.instantiate(cfg.test_token)
    test_tokenizers = {"test_tokenizer": test_tokenizer}
    tokenizers = fit_tokenizers | test_tokenizers

    if cfg.resume is not None and osp.isdir(cfg.resume):
        names = list(fit_tokenizers.keys())
        for name in names:
            fpath = osp.join(cfg.resume, f"{name}.pickle")
            if not osp.isfile(fpath):
                logging.error(
                    f"Cannot find pre-trained tokenizer '{name}'. (expected {osp.basename(fpath)} in {osp.dirname(fpath)})"
                )
                continue

            if cfg.verbose >= 1:
                logger.info(
                    f"Override AACTokenizer {name} with {fpath=}. (because resume is a result logdir)"
                )

            loaded_tokenizer = AACTokenizer.load(fpath)
            fit_tokenizers[name] = loaded_tokenizer

    # Build LightningDataModule with 'cfg.data'
    datamodule = hydra.utils.instantiate(cfg.data, **audio_transforms, **fit_tokenizers)

    # Build LightningModule with 'cfg.pl'
    pl_module = hydra.utils.instantiate(cfg.pl, **fit_tokenizers)

    # Build custom logger and callbacks
    callbacks = []
    pl_loggers = []

    if cfg.save:
        pl_logger = hydra.utils.instantiate(cfg.log)
        pl_loggers.append(pl_logger)
    else:
        pl_logger = None

    if cfg.trainer.enable_checkpointing:
        checkpoint = hydra.utils.instantiate(cfg.ckpt)
        callbacks.append(checkpoint)

    resume_callback = ResumeCallback(cfg.resume, cfg.verbose)
    callbacks.append(resume_callback)

    # Add callback for stop training if monitor is NaN
    early_stop_callback = EarlyStopping(
        monitor=cfg.ckpt.monitor, patience=sys.maxsize, check_finite=True
    )
    callbacks.append(early_stop_callback)

    # Add Evaluator for compute test metrics scores at the end of the training (when trainer.test is called)
    if cfg.evaluator is not None:
        evaluator = hydra.utils.instantiate(
            cfg.evaluator,
            test_tokenizer=test_tokenizer,
            verbose=cfg.verbose if cfg.tag != "NOTAG" else 0,
        )
        callbacks.append(evaluator)
    else:
        evaluator = None

    log_lr = LogLRCallback(bsize=cfg.data.bsize)
    callbacks.append(log_lr)

    log_grad_norm = LogGradNorm(bsize=cfg.data.bsize)
    callbacks.append(log_grad_norm)

    log_rng_state = LogRngState(bsize=cfg.data.bsize)
    callbacks.append(log_rng_state)

    if cfg.debug:
        log_gc = LogGCCallback(bsize=cfg.data.bsize)
        callbacks.append(log_gc)

    hydra_cfg = HydraConfig.get()
    test_mode = hydra_cfg.runtime.choices["test_mode"]
    on_end = ("test" if test_mode != "none" else "fit") if cfg.save else "none"

    stats_saver = StatsSaver(
        logdir=logdir,
        tokenizers=tokenizers,
        on_end=on_end,
        verbose=cfg.verbose,
    )
    callbacks.append(stats_saver)

    if test_mode == "swa":
        if datamodule is not None:
            datamodule.setup("fit")
        pl_module.setup("fit")

        swa_callback = hydra.utils.instantiate(cfg.test_mode.swa)
        callbacks.append(swa_callback)

    if cfg.debug:
        device_stats_monitor = DeviceStatsMonitor()
        callbacks.append(device_stats_monitor)

    if cfg.verbose >= 1:
        if cfg.tag != "NOTAG" or cfg.debug:
            max_depth = 20
        else:
            max_depth = 1

        model_summary = ModelSummary(max_depth=max_depth)
        callbacks.append(model_summary)

    trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=pl_loggers,
        callbacks=callbacks,
    )

    # --- START TRAINING
    trainer.fit(pl_module, datamodule=datamodule)

    # --- START TESTING
    checkpoint = trainer.checkpoint_callback
    if test_mode == "swa":
        if cfg.verbose >= 1:
            logger.info("Using SWA weights for testing...")
        trainer.test(pl_module, datamodule=datamodule, verbose=False)

    elif test_mode in ("best", "last_best"):
        if test_mode == "last_best":
            if cfg.verbose >= 1:
                logger.info("Test using last model...")
            stats_saver.set_active_save(False)
            if evaluator is not None:
                evaluator.set_prefix("last_ckpt_")
            trainer.test(pl_module, datamodule=datamodule, verbose=False)
            if evaluator is not None:
                evaluator.set_prefix("")
            stats_saver.set_active_save(True)

        if checkpoint is not None and osp.isfile(checkpoint.best_model_path):
            # Load best model before testing
            checkpoint_data = torch.load(
                checkpoint.best_model_path,
                map_location=pl_module.device,
            )
            pl_module.load_state_dict(checkpoint_data["state_dict"])
            if cfg.verbose >= 1:
                logger.info(
                    f"Test using best model {osp.basename(checkpoint.best_model_path)}..."
                )
        else:
            if cfg.verbose >= 1:
                logger.info(
                    f"Cannot find best model in checkpoint, use last weights for testing. ({test_mode=})"
                )
        trainer.test(pl_module, datamodule=datamodule, verbose=False)

    elif test_mode == "none":
        pass
    else:
        logging.error(
            f"Invalid argument {test_mode=}. (expected swa, best, last_best or none)"
        )

    if pl_logger is not None:
        out = pl_logger.metrics.get(cfg.out_crit, cfg.out_default)
    else:
        out = cfg.out_default

    if cfg.verbose >= 1:
        logger.info(f"Training is finished with {cfg.out_crit}={out}.")
    if cfg.verbose >= 1 and logdir is not None:
        logger.info(f"All results are saved in {logdir=}.")

    return out


if __name__ == "__main__":
    OmegaConf.register_new_resolver(name="include_keys", resolver=include_keys_func)
    main_train()
