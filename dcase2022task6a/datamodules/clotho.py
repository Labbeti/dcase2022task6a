#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from typing import Iterable, Optional, Sequence

from pytorch_lightning import LightningDataModule
from torch import nn
from torch.utils.data.dataloader import DataLoader

from dcase2022task6a.datamodules.utils import get_hdf_fpaths
from aac_datasets import Clotho
from dcase2022task6a.datasets.hdf import HDFDataset
from dcase2022task6a.nn.modules.misc import Lambda
from dcase2022task6a.tokenization import AACTokenizer, PAD_TOKEN, UNK_TOKEN
from dcase2022task6a.transforms.collate import PadCollateDict
from dcase2022task6a.transforms.text.select import SelectCaption


logger = logging.getLogger(__name__)


class ClothoDataModule(LightningDataModule):
    def __init__(
        self,
        # Datamodule common params
        root: str = "data",
        bsize: int = 8,
        n_workers: int = 0,
        pin_memory: bool = True,
        sr: int = 44100,
        verbose: int = 1,
        exclude_keys: Sequence[str] = (
            "captions_shape",
            "keywords",
            "keywords_shape",
            "sr",
        ),
        train_audio_transform: Optional[nn.Module] = None,
        val_audio_transform: Optional[nn.Module] = None,
        test_audio_transform: Optional[nn.Module] = None,
        fit_tokenizer: Optional[AACTokenizer] = None,
        # Other params
        version: str = "v2.1",
        download: bool = False,
        hdf_suffix: Optional[str] = None,
        test_subsets: Iterable[str] = Clotho.SUBSETS,
    ) -> None:
        """Init the Clotho datamodule for building dataloaders.

        :param root: The dataset parent directory, defaults to 'data'.
        :param bsize: The batch size of the dataloaders. defaults to 8.
        :param n_workers: The number of workers of the dataloaders. defaults to 0.
        :param pin_memory: If True, the dataloaders will pin memory of tensors. defaults to True.
        :param sr: The sample rate of the audio. defaults to 44100.
        :param verbose: If True, activate verbose. (default: True)
        :param exclude_keys: TODO
        :param train_audio_transform: The train audio transform to apply to each item. defaults to None.
        :param val_audio_transform: The val audio transform to apply to each item. defaults to None.
        :param test_audio_transform: The test audio transform to apply to each item. defaults to None.
        :param fit_tokenizer: The AACTokenizer for train and val captions. defaults to None.

        :param version: The version of the Clotho dataset. defaults to 'v2.1'.
        :param download: If True, download the dataset in the root directory. defaults to False.
        :param hdf_suffix: The filename suffix for the Clotho HDF dataset. defaults to None.
                If None, the default Clotho dataset will be used.
                Path: '{root}/HDF/clotho_{subset}_{hdf_suffix}.hdf'
        :param test_subsets: TODO
        """
        super().__init__()
        # Datamodule common params
        self._root = root
        self._bsize = bsize
        self._n_workers = n_workers
        self._pin_memory = pin_memory
        self._sr = sr
        self._verbose = verbose
        self._exclude_keys = exclude_keys
        self._train_audio_transform = train_audio_transform
        self._val_audio_transform = val_audio_transform
        self._test_audio_transform = test_audio_transform
        self._fit_tokenizer = fit_tokenizer
        # Other params
        self._version = version
        self._download = download
        self._hdf_suffix = hdf_suffix
        self._test_subsets = test_subsets

        clotho_subsets = Clotho.SUBSETS_DICT[self._version]
        if any(subset not in clotho_subsets for subset in test_subsets):
            raise ValueError(
                f"Invalid argument {test_subsets=}. (expected only subsets in {clotho_subsets})"
            )

        self._hdf_fpaths = get_hdf_fpaths(
            "clotho", clotho_subsets, self._root, self._hdf_suffix
        )

        self.save_hyperparameters(
            ignore=(
                "train_audio_transform",
                "val_audio_transform",
                "test_audio_transform",
                "fit_tokenizer",
            )
        )

    def prepare_data(self) -> None:
        if self._download:
            clotho_subsets = Clotho.SUBSETS_DICT[self._version]
            _ = [
                Clotho(
                    root=self._root,
                    subset=subset,
                    download=self._download,
                    version=self._version,
                )
                for subset in clotho_subsets
            ]

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            self._setup_fit()
        if stage in ("test", None):
            self._setup_test()

    def _setup_fit(self) -> None:
        train_subset = "dev"
        val_subset = "val" if self._version != "v1" else "eval"

        if isinstance(self._fit_tokenizer, AACTokenizer):
            encode_train = Lambda(
                self._fit_tokenizer.encode_rec,
                out_type="Tensor",
                add_sos_eos=True,
                unk_token=None,
            )
            encode_val = Lambda(
                self._fit_tokenizer.encode_rec,
                out_type="Tensor",
                add_sos_eos=True,
                unk_token=UNK_TOKEN,
            )
        else:
            encode_train = nn.Identity()
            encode_val = nn.Identity()

        self._train_captions_transform = nn.Sequential(
            SelectCaption(mode="random"),
            encode_train,
        )
        self._val_captions_transform = encode_val

        train_transforms = {
            "audio": self._train_audio_transform,
            "captions": self._train_captions_transform,
        }
        val_transforms = {
            "audio": self._val_audio_transform,
            "captions": self._val_captions_transform,
        }
        if self._hdf_suffix is None:
            self._train_dset = Clotho(
                root=self._root,
                subset=train_subset,
                transforms=train_transforms,
                version=self._version,
                item_type="dict",
            )
            self._val_dset = Clotho(
                root=self._root,
                subset=val_subset,
                transforms=val_transforms,
                version=self._version,
                item_type="dict",
            )
        else:
            if self._verbose >= 1:
                logger.debug(f"{self.__class__.__name__}: Setting up HDF datasets...")

            self._train_dset = HDFDataset(
                self._hdf_fpaths[train_subset],
                train_transforms,
                exclude_keys=self._exclude_keys,
            )
            self._val_dset = HDFDataset(
                self._hdf_fpaths[val_subset],
                val_transforms,
                exclude_keys=self._exclude_keys,
            )

        if (
            isinstance(self._fit_tokenizer, AACTokenizer)
            and not self._fit_tokenizer.is_fit()
        ):
            train_captions = [
                caption
                for dset in (self._train_dset,)
                for i in range(len(dset))
                for caption in dset.get_raw("captions", i)
            ]
            self._fit_tokenizer.fit(train_captions)

            if self._verbose >= 1:
                logger.debug(f"Vocabulary size: {self._fit_tokenizer.get_vocab_size()}")

        pad_values: dict = {"audio": 0.0}
        if isinstance(self._fit_tokenizer, AACTokenizer):
            pad_values["captions"] = self._fit_tokenizer.stoi(PAD_TOKEN)

        self._train_collate = PadCollateDict(
            pad_values, exclude_keys=self._exclude_keys
        )
        self._val_collate = PadCollateDict(pad_values, exclude_keys=self._exclude_keys)

    def _setup_test(self) -> None:
        pad_values = {"audio": 0.0}
        self._test_collate = PadCollateDict(pad_values, exclude_keys=self._exclude_keys)
        self._test_captions_transform = None

        test_subsets = self._test_subsets
        test_transforms = {
            "audio": self._test_audio_transform,
            "captions": self._test_captions_transform,
        }
        if self._hdf_suffix is None:
            self._test_dsets_lst = [
                Clotho(
                    root=self._root,
                    subset=subset,
                    transforms=test_transforms,
                    download=False,
                    version=self._version,
                    item_type="dict",
                )
                for subset in test_subsets
            ]
        else:
            self._test_dsets_lst = [
                HDFDataset(
                    self._hdf_fpaths[subset],
                    test_transforms,
                    exclude_keys=self._exclude_keys,
                )
                for subset in test_subsets
            ]

    def teardown(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            del self._train_dset
            del self._val_dset

        if stage in ("test", None):
            del self._test_dsets_lst
            self._test_dsets_lst = []

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self._train_dset,
            batch_size=self._bsize,
            num_workers=self._n_workers,
            shuffle=True,
            collate_fn=self._train_collate,
            pin_memory=self._pin_memory,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self._val_dset,
            batch_size=self._bsize,
            num_workers=self._n_workers,
            shuffle=False,
            collate_fn=self._val_collate,
            pin_memory=self._pin_memory,
            drop_last=False,
        )

    def test_dataloader(self) -> list[DataLoader]:
        return [
            DataLoader(
                dataset=dataset,
                batch_size=self._bsize,
                num_workers=self._n_workers,
                shuffle=False,
                collate_fn=self._test_collate,
                pin_memory=self._pin_memory,
                drop_last=False,
            )
            for dataset in self._test_dsets_lst
        ]
