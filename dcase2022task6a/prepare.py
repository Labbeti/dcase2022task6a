#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import math
import os
import os.path as osp
import pickle
import subprocess
import sys
import tqdm

from pathlib import Path
from subprocess import CalledProcessError
from typing import Any

import hydra
import nltk
import torch
import yaml

from gensim import downloader
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch import nn, Tensor
from torch.utils.data.dataset import Subset
from torchaudio.datasets.utils import download_url

from aac_datasets import AudioCaps
from aac_datasets import Clotho
from aac_datasets import MACS

from dcase2022task6a.datamodules.utils import get_hdf_fpaths
from dcase2022task6a.datasets.hdf import HDFDataset, pack_to_hdf
from dcase2022task6a.datasets.utils import AACDataset, filter_audio_sizes, AACSubset
from dcase2022task6a.metrics.bert_score import BertScore
from dcase2022task6a.metrics.bleu import Bleu
from dcase2022task6a.metrics.urls import JAR_URLS
from dcase2022task6a.nn.urls import PANN_PRETRAINED_URLS
from dcase2022task6a.tokenization import AACTokenizer
from dcase2022task6a.utils.ext import FUNCTION_WORDS_LINKS
from dcase2022task6a.utils.misc import any_checksum, reset_seed


logger = logging.getLogger(__name__)


def download_models(cfg: DictConfig) -> None:
    if cfg.nltk:
        # Download wordnet and omw-1.4 NLTK model for METEOR metric
        # Download punkt NLTK model for tokenizer
        for model_name in ("wordnet", "omw-1.4", "punkt", "averaged_perceptron_tagger"):
            nltk.download(model_name)

    if cfg.spacy:
        # Download spaCy model for AACTokenizer
        model = "en_core_web_sm"
        command = f"{sys.executable} -m spacy download {model}".split(" ")
        try:
            subprocess.check_call(
                command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            logger.info(f"Model '{model}' for spacy downloaded.")
        except (CalledProcessError, PermissionError) as err:
            logger.error(
                f"Cannot download spaCy model '{model}' for tokenizer. (command '{command}' with error={err})"
            )

    if cfg.bert:
        if cfg.verbose:
            logger.info("Downloading BERT models for BertScore...")
        _ = BertScore()

    if cfg.gensim:
        model_type = "word2vec-google-news-300"
        _ = downloader.load(model_type)

    if str(cfg.pann).lower() != "none":
        ckpt_dir = osp.join(torch.hub.get_dir(), "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        def can_download(name: str, pattern: Any) -> bool:
            if pattern == "all":
                return True
            elif isinstance(pattern, str):
                return name.lower() == pattern.lower()
            elif isinstance(pattern, list):
                return name.lower() in [pann_name.lower() for pann_name in pattern]
            elif isinstance(pattern, (bool, int)):
                return can_download(name, "all" if pattern else "none")
            else:
                raise TypeError(
                    f"Invalid cfg.pann argument. Must be a string, a list of strings, a bool or an int, found {pattern.__class__.__name__}."
                )

        urls = {
            name: model_info
            for name, model_info in PANN_PRETRAINED_URLS.items()
            if can_download(name, cfg.pann)
        }

        for i, (name, model_info) in enumerate(urls.items()):
            path = osp.join(ckpt_dir, model_info["fname"])

            if osp.isfile(path):
                logger.info(
                    f"Model '{name}' already downloaded in '{path}'. ({i+1}/{len(urls)})"
                )
            else:
                logger.info(
                    f"Start downloading pre-trained PANN model '{name}' ({i+1}/{len(urls)})..."
                )
                torch.hub.download_url_to_file(model_info["url"], path)
                logger.info(f"Model '{name}' downloaded in '{path}'.")


def download_dataset(cfg: DictConfig) -> dict[str, AACDataset]:
    # Download a dataset
    hydra_cfg = HydraConfig.get()
    dataname = hydra_cfg.runtime.choices["data"]

    dsets: dict[str, AACDataset]

    if dataname == "clotho":
        subsets = Clotho.SUBSETS_DICT[cfg.data.version]
        dsets = {
            subset: Clotho(
                root=cfg.data.root,
                subset=subset,
                version=cfg.data.version,
                download=cfg.data.download,
                verbose=cfg.verbose,
                item_type="dict",
            )
            for subset in subsets
        }

    else:
        raise RuntimeError(
            f"Unknown dataset '{cfg.dataset}'. Must be 'clotho'."
        )

    min_audio_size = float(cfg.datafilter.min_audio_size)
    max_audio_size = float(cfg.datafilter.max_audio_size)
    if min_audio_size > 0.0 or not math.isinf(max_audio_size):
        for subset in dsets.keys():
            dset = dsets[subset]
            indexes = filter_audio_sizes(
                dset, min_audio_size, max_audio_size, cfg.verbose
            )
            dsets[subset] = AACSubset(dset, indexes)

    return dsets


def pack_dsets_to_hdf(cfg: DictConfig, dsets: dict[str, Any]) -> None:
    if not cfg.pack_to_hdf:
        return

    hydra_cfg = HydraConfig.get()
    dataname = hydra_cfg.runtime.choices["data"]
    audio_transform_name = hydra_cfg.runtime.choices["audio_trans"]
    sentence_transform_name = hydra_cfg.runtime.choices["text_trans"]

    if len(dsets) == 0:
        logger.warning(
            f"Invalid value {dataname=} for pack_to_hdf=true. (expected one of 'audiocaps', 'clotho', 'macs')"
        )

    for dset in dsets.values():
        if hasattr(dset, "SAMPLE_RATE"):
            dset_sr = dset.SAMPLE_RATE
            if dset_sr != cfg.data.sr:
                raise ValueError(
                    f"Invalid {cfg.data.sr=}. (expected {dset_sr} for {dataname})"
                )

    if cfg.debug_lim_dset is not None:
        if cfg.verbose >= 1:
            logger.info(f"Limit datasets to {cfg.debug_lim_dset}.")

        dsets = {
            subset: Subset(dset, list(range(min(cfg.debug_lim_dset, len(dset)))))
            for subset, dset in dsets.items()
        }

    hdf_root = osp.join(cfg.path.data, "HDF")
    os.makedirs(hdf_root, exist_ok=True)

    for subset, dset in dsets.items():
        audio_transform_params = dict(cfg.audio_trans)
        sentence_transform_params = dict(cfg.text_trans)

        audio_transform = hydra.utils.instantiate(audio_transform_params)
        sentence_transform = hydra.utils.instantiate(sentence_transform_params)
        pre_save_transforms = {
            "audio": audio_transform,
            "captions": sentence_transform,
        }
        transforms_params = {
            "audio": audio_transform_params,
            "captions": sentence_transform_params,
        }
        csum = any_checksum(transforms_params, 1000)
        hdf_fname = f"{dataname}_{subset}_{audio_transform_name}_{sentence_transform_name}_{csum}.hdf"
        hdf_fpath = osp.join(hdf_root, hdf_fname)

        if not osp.isfile(hdf_fpath) or cfg.overwrite_hdf:
            if cfg.verbose:
                logger.info(
                    f"Start packing the {dataname}_{subset} dataset to HDF file {hdf_fname}..."
                )

            metadata = {
                "transform_params": transforms_params,
            }
            if hasattr(cfg.audio_trans, "tgt_sr"):
                metadata["sr"] = cfg.audio_trans.tgt_sr

            if cfg.verbose:
                logger.debug(yaml.dump({"Metadata": metadata}))

            hdf_dset = pack_to_hdf(
                dset,
                hdf_fpath,
                pre_save_transforms,
                overwrite=cfg.overwrite_hdf,
                metadata=str(metadata),
            )
        else:
            if cfg.verbose:
                logger.info(
                    f"Dataset {dataname}_{subset} is already packed to hdf in {hdf_fpath=}."
                )

            hdf_dset = HDFDataset(hdf_fpath)

        if cfg.debug:
            # Sanity check
            idx = int(torch.randint(len(dset), ()).item())

            dset_item = dict(dset[idx])
            for name, transform in pre_save_transforms.items():
                if name in dset_item.keys() and transform is not None:
                    dset_item[name] = transform(dset_item[name])
            hdf_item = hdf_dset[idx]

            dset_keys_in_hdf_keys = all(
                key in hdf_item.keys() for key in dset_item.keys()
            )
            same_dset_len = len(dset) == len(hdf_dset)

            logger.debug(f"Check with item N°{idx=}")
            logger.debug(
                f"Check {dset_keys_in_hdf_keys=} ({dset_item.keys()} in {hdf_item})"
            )
            logger.debug(f"Check {same_dset_len=} ({len(dset)} == {len(hdf_dset)})")

            all_same = True

            if "audio" in dset_item.keys():
                rtol = 10 ** -3
                dset_audio, hdf_audio = dset_item["audio"], hdf_item["audio"]
                same_audio_shape = dset_audio.shape == hdf_audio.shape
                close_audio = same_audio_shape and torch.allclose(
                    dset_audio, hdf_audio, rtol=rtol
                )
                same_audio = same_audio_shape and dset_audio.eq(hdf_audio).all().item()
                all_same = all_same and close_audio and same_audio

                logger.debug(
                    f"Check {same_audio_shape=} ({dset_audio.shape} == {hdf_audio.shape})"
                )
                logger.debug(f"Check {close_audio=} ({rtol=})")
                logger.debug(f"Check {same_audio=}")

            if "captions" in dset_item.keys():
                dset_captions, hdf_captions = (
                    dset_item["captions"],
                    hdf_item["captions"],
                )
                same_captions = len(dset_captions) == len(hdf_captions) and all(
                    c1 == c2 for c1, c2 in zip(dset_captions, hdf_captions)
                )
                captions_eq = (
                    f"(\n{dset_captions}\n == \n{hdf_captions}\n)"
                    if not same_captions
                    else ""
                )
                all_same = all_same and same_captions

                logger.debug(f"Check {same_captions=} {captions_eq}")

            if not all_same:
                logger.warning(
                    f"Check has failed after packing {dataname} to HDF. (dataset={dset.__class__.__name__}, {subset=})\n"
                    f"NOTE: if a transform is stochastic, you can ignore this warning."
                )


def download_other(cfg: DictConfig) -> None:
    meta_dpath = osp.join(cfg.path.data, "meta")
    os.makedirs(meta_dpath, exist_ok=True)

    if cfg.audioset_indices:
        url = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
        fname = "class_labels_indices.csv"
        fpath = osp.join(meta_dpath, fname)
        if not osp.isfile(fpath):
            logger.info(f"Downloading file {fname}...")
            download_url(url, meta_dpath, fname, progress_bar=cfg.verbose)

    if cfg.function_words:
        func_words_dpath = osp.join(meta_dpath, "function_words")
        os.makedirs(func_words_dpath, exist_ok=True)

        for name, url in FUNCTION_WORDS_LINKS.items():
            fname = f"{name}.dat"
            if not osp.isfile(osp.join(func_words_dpath, fname)):
                logger.info(f"Downloading file {fname}...")
                download_url(url, func_words_dpath, fname, progress_bar=cfg.verbose)

    if cfg.stanford_nlp:
        stanford_nlp_dpath = osp.join(cfg.path.ext, "stanford_nlp")
        os.makedirs(stanford_nlp_dpath, exist_ok=True)

        name = "stanford_nlp"
        info = JAR_URLS[name]
        url = info["url"]
        fname = info["fname"]
        fpath = osp.join(stanford_nlp_dpath, fname)
        if not osp.isfile(fpath):
            if cfg.verbose:
                logger.info(
                    f"Downloading jar source for '{name}' in directory {stanford_nlp_dpath}."
                )
            download_url(url, stanford_nlp_dpath, fname, progress_bar=cfg.verbose)

    if cfg.meteor:
        meteor_dpath = osp.join(cfg.path.ext, "meteor")
        os.makedirs(meteor_dpath, exist_ok=True)

        for name in ("meteor", "meteor_data"):
            info = JAR_URLS[name]
            url = info["url"]
            fname = info["fname"]
            subdir = osp.dirname(fname)
            fpath = osp.join(meteor_dpath, fname)

            if not osp.isfile(fpath):
                if cfg.verbose:
                    logger.info(
                        f"Downloading jar source for '{name}' in directory {meteor_dpath}."
                    )
                if subdir not in ("", "."):
                    os.makedirs(osp.join(meteor_dpath, subdir), exist_ok=True)
                download_url(
                    url, meteor_dpath, fname, progress_bar=cfg.verbose, resume=True
                )

    if cfg.spice:
        script = osp.join(Path(__file__).parent, "..", "install_spice.sh")
        if not osp.isfile(script):
            raise FileNotFoundError(f"Cannot find script '{script}'.")
        spice_dpath = osp.join(cfg.path.ext, "spice")
        os.makedirs(spice_dpath, exist_ok=True)

        command = ["bash", script, spice_dpath]
        try:
            subprocess.check_call(
                command,
                stdout=None if cfg.verbose else subprocess.DEVNULL,
                stderr=None if cfg.verbose else subprocess.DEVNULL,
            )
        except (subprocess.CalledProcessError, PermissionError) as err:
            logger.error(err)

    if cfg.words:
        url = "https://github.com/dwyl/english-words/raw/master/words_alpha.txt"
        fname = "words_alpha.txt"
        fpath = osp.join(meta_dpath, fname)

        if not osp.isfile(fpath):
            download_url(
                url,
                meta_dpath,
                fname,
                progress_bar=cfg.verbose,
                resume=True,
            )


@hydra.main(config_path=osp.join("..", "conf"), config_name="prepare")
def main_prepare(cfg: DictConfig) -> None:
    """Download models and datasets."""
    reset_seed(cfg.seed)
    if cfg.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if cfg.verbose >= 1 and (cfg.tag != "NOTAG" or cfg.debug):
        logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
        logger.info(f"Datetime: {cfg.datetime}\n")

    if cfg.path.torch_hub is not None:
        os.makedirs(cfg.path.torch_hub, exist_ok=True)
        torch.hub.set_dir(cfg.path.torch_hub)

    download_models(cfg)
    dsets = download_dataset(cfg)
    pack_dsets_to_hdf(cfg, dsets)
    download_other(cfg)

    if cfg.verbose >= 1:
        logdir = osp.join(cfg.log.save_dir, cfg.log.name, cfg.log.version)
        logger.info(f'All preparation logs are saved in logdir "{logdir}".')


if __name__ == "__main__":
    main_prepare()
