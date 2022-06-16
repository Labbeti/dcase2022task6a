#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import logging
import os.path as osp
import time

from typing import Any, Iterable, Optional, Sequence

import torch
import yaml

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
from torch import Tensor

from dcase2022task6a.callbacks.utils import get_pl_loggers
from dcase2022task6a.metrics.all import WordLevelMetrics, SentenceLevelMetrics
from dcase2022task6a.tokenization import AACTokenizer
from dcase2022task6a.utils.custom_logger import CustomTensorboardLogger
from dcase2022task6a.utils.misc import all_eq


logger = logging.getLogger(__name__)


class CaptioningEvaluator(Callback):
    """
    Callback that store predictions and captions during testing for produce AAC scores
    (BLEU1, BLEU2, BLEU3, BLEU4, METEOR, ROUGE-L, CIDEr, SPICE, SPIDEr).
    """

    def __init__(
        self,
        logdir: Optional[str],
        test_tokenizer: AACTokenizer,
        java_path: str = "java",
        ext_dpath: str = "ext",
        tmp_dpath: str = "/tmp",
        prefix: str = "",
        verbose: int = 1,
        debug: bool = False,
        save_to_csv: bool = True,
        save_dcase_csv_file: bool = False,
    ) -> None:
        super().__init__()
        self._logdir = logdir
        self._test_tokenizer = test_tokenizer
        self._java_path = java_path
        self._ext_path = ext_dpath
        self._tmp_path = tmp_dpath
        self._prefix = prefix
        self._verbose = verbose
        self._debug = debug
        self._save_to_csv = save_to_csv
        self._save_dcase_csv_file = save_dcase_csv_file

        self._all_outputs = {}

        # Note : we avoid compute scores for
        # - AudioCaps/train because it is too large
        # - Clotho/test because it does not have any references
        # - Clotho/anasysis because it does not have any references
        self._excluded_datasubsets_metrics = (
            "audiocaps_train",
            "clotho_test",
            "clotho_analysis",
        )
        self._cap_key = "captions"

    def set_prefix(self, prefix: str) -> None:
        self._prefix = prefix

    def on_fit_end(self, trainer, pl_module: LightningModule) -> None:
        return super().on_fit_end(trainer, pl_module)

    def on_test_start(self, trainer, pl_module) -> None:
        self._sent_level_metrics = SentenceLevelMetrics(
            java_path=self._java_path,
            ext_dpath=self._ext_path,
            tmp_dpath=self._tmp_path,
            check_input=self._debug,
            device=pl_module.device,
            verbose=self._verbose,
        )
        self._word_level_metrics = WordLevelMetrics(
            check_input=self._debug,
            verbose=self._verbose,
        )

    def on_test_epoch_start(self, trainer, pl_module) -> None:
        self._all_outputs = {}
        if self._verbose >= 1:
            logger.debug(f"Starting test epoch with prefix='{self._prefix}'")

    def on_test_batch_end(
        self,
        trainer,
        pl_module,
        outputs: dict[str, Any],
        batch: dict[str, Any],
        batch_idx,
        dataloader_idx: int,
    ) -> None:
        if outputs is None:
            logging.warning(
                f"Lightning module has returned None at test step {batch_idx}."
            )
            return None

        if dataloader_idx not in self._all_outputs.keys():
            self._all_outputs[dataloader_idx] = {}

        # Get output values
        for key, batch_values in outputs.items():
            if key not in self._all_outputs[dataloader_idx].keys():
                self._all_outputs[dataloader_idx][key] = []

            if isinstance(batch_values, Iterable):
                if (
                    isinstance(batch_values, Sequence)
                    and len(batch_values) > 0
                    and isinstance(batch_values[0], Tensor)
                ):
                    batch_values = [v.cpu() for v in batch_values]
                self._all_outputs[dataloader_idx][key] += list(batch_values)
            else:
                logging.warning(
                    f"Skipping batch values with {key=}. (values are not iterable)"
                )

        # Get batch values
        for key in ("fname", "index", "dataset", "subset"):
            if key not in batch.keys():
                raise ValueError(f"Cannot find {key=} in batch.")
            if key not in self._all_outputs[dataloader_idx].keys():
                self._all_outputs[dataloader_idx][key] = []
            self._all_outputs[dataloader_idx][key] += batch[key]

    def on_test_epoch_end(self, trainer, pl_module: LightningModule) -> None:
        datasubsets = []
        for outputs in self._all_outputs.values():
            # Sanity check
            n_items = len(next(iter(outputs.values())))
            invalid_sizes_keys = [
                key for key, values in outputs.items() if len(values) != n_items
            ]
            if len(invalid_sizes_keys) > 0:
                raise RuntimeError(
                    f"Invalid number of values for keys={invalid_sizes_keys} (expected {n_items} but found {[len(outputs[key]) for key in invalid_sizes_keys]})."
                )

            datanames = list(sorted(set(map(str.lower, outputs["dataset"]))))
            subsets = list(sorted(set(map(str.lower, outputs["subset"]))))
            if len(datanames) == 1 and len(subsets) == 1:
                datasubset = f"{datanames[0]}_{subsets[0]}"
            else:
                datasubset = f"mix_{'_'.join(datanames)}_{'_'.join(subsets)}"

            counter = datasubsets.count(datasubset)
            if counter > 0:
                old_datasubset = datasubset
                datasubset = f"{datasubset}_{counter+1}"
                logging.error(
                    f"Found duplicated subset '{old_datasubset}'. Renaming to '{datasubset}'."
                )
                assert datasubset not in datasubsets
            datasubsets.append(datasubset)

            # Tokenize hypotheses and references
            keys = [
                key
                for key in outputs
                if key.startswith("preds") or key == self._cap_key
            ]

            def clean_sentences(sentences: list[str]) -> list[str]:
                encoded = self._test_tokenizer.encode_batch(sentences)
                decoded = self._test_tokenizer.decode_batch(encoded)
                return decoded

            for key in keys:
                all_caps = outputs[key]
                if not isinstance(all_caps, list):
                    raise TypeError(
                        f"Invalid output sentences type. (found {type(all_caps)} but expected list)"
                    )

                # If list[str]
                if all(isinstance(caps, str) for caps in all_caps):
                    all_caps = clean_sentences(all_caps)
                # If list[list[str]]
                elif all(isinstance(caps, list) for caps in all_caps) and all(
                    isinstance(cap, str) for caps in all_caps for cap in caps
                ):
                    # We need to flat sentences for making a single call to encode_batch because PTB Tokenizer is really slow.
                    caps_counts = [len(caps) for caps in all_caps]
                    all_caps_flat = [cap for caps in all_caps for cap in caps]
                    all_caps_flat = clean_sentences(all_caps_flat)
                    all_caps = []
                    i = 0
                    for count in caps_counts:
                        all_caps.append(all_caps_flat[i : i + count])
                        i += count
                else:
                    raise TypeError(
                        "Invalid output sentences type. (expected list[str] or list[list[str]])"
                    )

                outputs[key] = all_caps

            local_scores = {}
            if datasubset not in self._excluded_datasubsets_metrics:
                global_scores, local_scores = self.compute_metrics(outputs, datasubset)
                self.log_global_scores(global_scores, datasubset, pl_module)
                if self._verbose >= 1:
                    self.print_example(outputs, datasubset)
            else:
                logger.debug(f"Skipping metrics for subset '{datasubset}'...")

            if self._save_to_csv:
                if self._logdir is not None and osp.isdir(self._logdir):
                    self.save_outputs_to_csv(
                        self._logdir,
                        datasubset,
                        outputs,
                        local_scores,
                    )
                else:
                    logging.error(
                        f"Cannot save outputs to CSV because logdir is not a valid directory. (logdir={self._logdir}, {datasubset=})"
                    )

    def compute_metrics(
        self,
        outputs: dict[str, list],
        datasubset: str,
    ) -> tuple[dict[str, dict], dict[str, Tensor]]:
        all_caps_sents = outputs[self._cap_key]
        word_tokenizer = AACTokenizer(level="word", backend="python")
        all_caps_words = word_tokenizer(all_caps_sents)

        start_time = time.perf_counter()
        if self._verbose >= 1:
            n_metrics = 0
            if self._word_level_metrics is not None:
                n_metrics += len(self._word_level_metrics)
            if self._sent_level_metrics is not None:
                n_metrics += len(self._sent_level_metrics)
            logger.info(
                f"Start computing metrics... ({datasubset=}, n_outputs={len(outputs)}, {n_metrics=})"
            )

        global_scores = {}
        local_scores = {}
        pred_keys = [
            key
            for key, values in outputs.items()
            if key.startswith("preds")
            and isinstance(values, list)
            and all(isinstance(value, str) for value in values)
        ]

        for pred_key in pred_keys:
            all_hyps_sents = outputs[pred_key]
            global_scores[pred_key] = {}

            if self._word_level_metrics is not None:
                if self._verbose >= 1:
                    logger.debug(
                        f"Computing word level metrics... ({datasubset=}, {pred_key=})"
                    )

                all_hyps_words = word_tokenizer(all_hyps_sents)
                global_scores[pred_key] |= self._word_level_metrics(
                    all_hyps_words,
                    all_caps_words,
                )

            if self._sent_level_metrics is not None:
                if self._verbose >= 1:
                    logger.debug(
                        f"Computing sentence level metrics... ({datasubset=}, {pred_key=})"
                    )

                sent_scores = self._sent_level_metrics(
                    all_hyps_sents,
                    all_caps_sents,
                    return_dict=True,
                )
                local_scores |= {
                    f"{metric_name}_{pred_key}": scores["scores"]
                    for metric_name, scores in sent_scores.items()
                }
                global_scores[pred_key] |= {
                    metric_name: scores["score"]
                    for metric_name, scores in sent_scores.items()
                }

        end_time = time.perf_counter()
        if self._verbose >= 1:
            duration_s = end_time - start_time
            logger.info(
                f"Computing metrics finished in {duration_s:.2f}s. ({datasubset=})"
            )

        return global_scores, local_scores

    def log_global_scores(
        self,
        scores: dict[str, dict[str, Any]],
        datasubset: str,
        pl_module: LightningModule,
    ) -> None:
        flatten_scores = {
            f"{self._prefix}{datasubset}/{metric_name}_{pred_key}": score
            for pred_key, pred_scores in scores.items()
            for metric_name, score in pred_scores.items()
        }

        pl_loggers = get_pl_loggers(pl_module)
        for pl_logger in pl_loggers:
            if isinstance(pl_logger, CustomTensorboardLogger):
                if self._verbose >= 1:
                    logger.info(
                        f"Saving scores for dataset {datasubset}:\n{yaml.dump(scores, sort_keys=False)}"
                    )
                pl_logger.log_hyperparams(params={}, metrics=flatten_scores)

    def print_example(self, outputs: dict[str, list], datasubset: str) -> None:
        assert self._test_tokenizer is not None
        n_outputs = len(outputs["fname"])
        indexes = torch.randint(0, n_outputs, (1,)).tolist()

        logger.info(f"Show {len(indexes)} example(s) : ")
        for idx in indexes:
            fname = outputs["fname"][idx]
            dset_index = outputs["index"][idx]
            hypotheses = {
                key: hypotheses_sents[idx]
                for key, hypotheses_sents in outputs.items()
                if key.startswith("preds")
            }
            references = outputs[self._cap_key][idx]

            lines = "-" * 12
            width = 128
            logger.info(
                f"Output N°{idx}/{n_outputs} with evaluator_prefix='{self._prefix}' ({datasubset=}, {fname=}, {dset_index=})\n"
                f"{lines}\nHypotheses :\n{lines}\n{yaml.dump(hypotheses, width=width, sort_keys=False)}"
                f"{lines}\nReferences :\n{lines}\n{yaml.dump(references, width=width, sort_keys=False)}"
            )

    def save_outputs_to_csv(
        self,
        dpath: str,
        datasubset: str,
        outs: dict[str, list],
        local_scores: dict[str, Tensor],
    ) -> None:
        lens = list(map(len, outs.values()))
        assert all_eq(lens)

        n_items = lens[0]
        fname = f"{self._prefix}outputs_{datasubset}.csv"
        fpath = osp.join(dpath, fname)

        def process(key: str, value: Any) -> Any:
            if isinstance(value, (Tensor,)):
                return value.tolist()
            else:
                return value

        csv_all_values = outs | local_scores

        with open(fpath, "w") as file:
            keys = list(csv_all_values.keys())
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()

            for i in range(n_items):
                row = {key: values[i] for key, values in csv_all_values.items()}
                row = {key: process(key, value) for key, value in row.items()}
                writer.writerow(row)

        if self._save_dcase_csv_file:
            fnames = outs["fname"]
            preds = outs["preds"]
            datadict = {"file_name": fnames, "caption_predicted": preds}
            datalist = [{k: v[i] for k, v in datadict.items()} for i in range(n_items)]

            fname = f"submission_output_{datasubset}.csv"
            fpath = osp.join(dpath, fname)
            with open(fpath, "w") as file:
                keys = list(datadict.keys())
                writer = csv.DictWriter(file, fieldnames=keys)
                writer.writeheader()
                writer.writerows(datalist)
