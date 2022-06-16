#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import math
import pickle
import tqdm

from functools import cache
from typing import (
    Any,
    Callable,
    Iterable,
    Optional,
    Protocol,
    runtime_checkable,
    Sized,
    Union,
)

import torch
import torchaudio

from torch import nn
from torch.utils.data.dataset import Dataset


@runtime_checkable
class SizedDataset(Protocol):
    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError("Protocal abstract method.")

    def __len__(self) -> int:
        raise NotImplementedError("Protocal abstract method.")


@runtime_checkable
class AACDataset(Protocol):
    """Protocal abstract class for aac datasets. Used only for typing.

    Methods signatures:
        - __len__: () -> int
        - __getitem__: int -> Any
        - get: (str, int) -> Any
        - get_raw: (str, int) -> Any
    """

    def __len__(self) -> int:
        raise NotImplementedError("Protocal abstract method.")

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError("Protocal abstract method.")

    def get(self, name: str, index: int) -> Any:
        raise NotImplementedError("Protocal abstract method.")

    def get_raw(self, name: str, index: int) -> Any:
        raise NotImplementedError("Protocal abstract method.")


class EmptyDataset(Dataset):
    def __getitem__(self, index: Any) -> None:
        raise NotImplementedError(
            f"Invalid call of getitem for {self.__class__.__name__}."
        )

    def __len__(self) -> int:
        return 0


class LambdaDataset(Dataset):
    def __init__(
        self,
        func: Callable[[int], Any],
        length: int,
        func_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self._func = func
        self._length = length
        self._func_kwargs = func_kwargs if func_kwargs is not None else {}

    def __getitem__(self, index: int) -> Any:
        return self._func(index, **self._func_kwargs)

    def __len__(self) -> int:
        return self._length


class Wrapper(Dataset):
    """
    Base class for dataset wrappers.

    Can recursively get an attribute of the dataset wrapped :

    >>> audiocaps = AudioCaps(...) # has method 'get()'
    >>> audiocaps_wrapped = Wrapper(audiocaps)
    >>> audiocaps.get(...) # OK
    >>> audiocaps_wrapped.get(...) # OK

    :param source: The source dataset to wrap.
    :param recursive_getattr: If True, recursively check if the wrapped dataset 'source' has an attribute when getattribute is called.
    """

    def __init__(self, source: Any, recursive_getattr: bool = False) -> None:
        super().__init__()
        self._source = source
        self._recursive_getattr = recursive_getattr

    def unwrap(self, recursive: bool = True) -> Any:
        if not recursive:
            return self._source
        else:
            dset = self._source
            while isinstance(dset, Wrapper):
                dset = dset.unwrap()
            return dset

    def __getitem__(self, index: int) -> Any:
        return self._source.__getitem__(index)

    def __len__(self) -> int:
        if isinstance(self._source, Sized):
            return len(self._source)
        else:
            raise NotImplementedError(
                f"Wrapped dataset {self._source.__class__.__name__} is not Sized."
            )

    def __getattribute__(self, name: str) -> Any:
        try:
            return super().__getattribute__(name)
        except AttributeError:
            pass

        message = (
            f"Wrapper {self.__class__.__name__} does not have the attribute '{name}'."
        )
        if not self._recursive_getattr:
            raise AttributeError(message)
        else:
            try:
                return self._source.__class__.__getattribute__(self, name)
            except AttributeError:
                raise AttributeError(message)


class AACSubset(Wrapper):
    """Similar to torch.utils.data.Subset but for AACDataset classes."""

    def __init__(self, dataset: AACDataset, indexes: Iterable[int]) -> None:
        super().__init__(dataset, recursive_getattr=False)
        self._indexes = list(indexes)

    def get_raw(self, name: str, index: int) -> Any:
        local_index = self._indexes[index]
        return self._source.get_raw(name, local_index)

    def get(self, name: str, index: int) -> Any:
        local_index = self._indexes[index]
        return self._source.get(name, local_index)

    def __getitem__(self, index: int) -> Any:
        local_index = self._indexes[index]
        return self._source[local_index]

    def __len__(self) -> int:
        return len(self._indexes)


class AACConcat(Wrapper):
    """Similar to torch.utils.data.ConcatDataset but for AACDataset classes."""

    def __init__(self, *datasets: AACDataset) -> None:
        super().__init__(datasets, recursive_getattr=False)
        cumsum = []
        prev_size = 0
        for dset in datasets:
            dset_size = len(dset)
            cumsum.append(dset_size + prev_size)
            prev_size = dset_size
        self._cumsum = cumsum
        assert self._cumsum[-1] == len(
            self
        ), f"Found {self._cumsum[-1]=} != {len(self)=}."

    @cache
    def _index_to_dset_and_local_indexes(self, index: int) -> tuple[int, int]:
        if index < 0 or index >= self._cumsum[-1]:
            raise IndexError(f"Invalid index {index} for {self.__class__.__name__}.")

        local_index = None
        dset_idx = None
        prevsum = 0
        for i, sum_ in enumerate(self._cumsum):
            if index < sum_:
                dset_idx = i
                local_index = index - prevsum
                break
            prevsum = sum_
        if local_index is None or dset_idx is None:
            raise IndexError(
                f"Invalid index {index} for {self.__class__.__name__}. (found {local_index=} and {dset_idx=})"
            )

        return dset_idx, local_index

    def get_raw(self, name: str, index: int) -> Any:
        dset_idx, local_index = self._index_to_dset_and_local_indexes(index)
        return self._source[dset_idx].get_raw(name, local_index)

    def get(self, name: str, index: int) -> Any:
        dset_idx, local_index = self._index_to_dset_and_local_indexes(index)
        return self._source[dset_idx].get(name, local_index)

    def __getitem__(self, index: int) -> Any:
        dset_idx, local_index = self._index_to_dset_and_local_indexes(index)
        return self._source[dset_idx][local_index]

    def __len__(self) -> int:
        return sum(map(len, self._source))


class PostTransformWrap(Wrapper):
    def __init__(
        self,
        dataset: Dataset,
        transform: Optional[Callable],
        index: Union[None, int, str] = None,
    ) -> None:
        super().__init__(dataset)
        self._transform = transform
        self._index = index

    def __getitem__(self, index: int) -> Any:
        item = self._source.__getitem__(index)
        if self._transform is None:
            return item
        elif self._index is None:
            return self._transform(item)
        else:
            if isinstance(item, tuple):
                return tuple(
                    (self._transform(sub_item) if i == self._index else sub_item)
                    for i, sub_item in enumerate(item)
                )
            elif isinstance(item, dict):
                item[self._index] = self._transform(item[self._index])
            else:
                raise TypeError(
                    f"Invalid item type {type(item)}. (expected tuple or dict)"
                )


class CacheWrap(Wrapper):
    def __init__(self, dataset: Any) -> None:
        super().__init__(dataset)

    @cache
    def __getitem__(self, index: int) -> tuple:
        return self._source.__getitem__(index)

    @cache
    def __len__(self) -> int:
        return len(self._source)

    def load_items(
        self, verbose: bool = False, desc: str = "Loading dataset..."
    ) -> None:
        for i in tqdm.trange(len(self), disable=not verbose, desc=desc):
            self[i]


class Duplicate(Wrapper):
    def __init__(self, dataset: Dataset, n_samples_max: int) -> None:
        assert isinstance(dataset, Sized)
        assert len(dataset) <= n_samples_max
        super().__init__(dataset)
        self._n_samples_max = n_samples_max

    def __getitem__(self, index: int) -> Any:
        local_index = index % len(self._source)
        return self._source[local_index]

    def __len__(self) -> int:
        return self._n_samples_max


class CaptionsFilter(Wrapper):
    CAPTIONS_KEY = "captions"

    def __init__(
        self,
        dataset: AACDataset,
        indexes_dic: dict[int, list[int]],
        post_transform: Optional[nn.Module] = None,
    ) -> None:
        for method in ("get_raw", "get", "__len__", "__getitem__"):
            if not hasattr(dataset, method):
                raise ValueError(
                    f"Invalid argument dataset={dataset.__class__.__name__} for {self.__class__.__name__}."
                    f'(dataset argument does not have a "{method}" method).'
                )
        # Sanity check
        for cloidx in indexes_dic.keys():
            if not (0 <= cloidx < len(dataset)):
                raise ValueError(
                    f"Invalid value indexes_dic with {cloidx=} with dataset={dataset}."
                )
        super().__init__(dataset, recursive_getattr=True)
        self._dic_indexes_idc = dict(enumerate(indexes_dic.items()))
        self._transforms = {self.CAPTIONS_KEY: post_transform}

    def get_raw(self, name: str, index: int) -> Any:
        if index not in self._dic_indexes_idc.keys():
            raise ValueError(f"Invalid value {index=} for {self.__class__.__name__}.")

        cloidx, capindexes = self._dic_indexes_idc[index]
        value = self._source.get_raw(name, cloidx)
        if name == self.CAPTIONS_KEY:
            value = [value[capidx] for capidx in capindexes]
        return value

    def get(self, name: str, index: int) -> Any:
        value = self.get_raw(name, index)

        if hasattr(self._source, "_transforms"):
            transform = self._source._transforms.get(name, None)
            if transform is not None:
                value = transform(value)

        transform = self._transforms.get(name, None)
        if transform is not None:
            value = transform(value)
        return value

    def __getitem__(self, index: int) -> Any:
        cloidx, _ = self._dic_indexes_idc[index]
        item = self._source.__getitem__(cloidx)
        item[self.CAPTIONS_KEY] = self.get(self.CAPTIONS_KEY, index)
        return item

    def __len__(self) -> int:
        return len(self._dic_indexes_idc)

    @classmethod
    def from_pickle(
        cls, dataset, fpath: str, captions_transform: Optional[nn.Module] = None
    ) -> "CaptionsFilter":
        with open(fpath, "rb") as file:
            indexes_dic = pickle.load(file)
            return CaptionsFilter(dataset, indexes_dic, captions_transform)

    @classmethod
    def from_json(
        cls, dataset, fpath: str, captions_transform: Optional[nn.Module] = None
    ) -> "CaptionsFilter":
        with open(fpath, "r") as file:
            indexes_dic = json.load(file)
            return CaptionsFilter(dataset, indexes_dic, captions_transform)

    @classmethod
    def from_fpath(
        cls, dataset, fpath: str, captions_transform: Optional[nn.Module] = None
    ) -> "CaptionsFilter":
        if fpath.endswith(".json"):
            return cls.from_json(dataset, fpath, captions_transform)
        elif fpath.endswith(".pickle"):
            return cls.from_pickle(dataset, fpath, captions_transform)
        else:
            raise ValueError(
                f'Invalid extension {fpath.split(".")[-1]}. (expected json or pickle)'
            )


class DsetTestSample(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self._all_captions = [
            (
                "Cars travel past in the distance as a clock ticks",
                "A clock is ticking with traffic in the background",
                "An old clock with a pendulum that is swinging back and forth is ticking.",
                "An old clock with a pendulum is ticking.",
                "The industrial time card clock emits a thick, hallow ticking.",
            ),
            (
                "Chicks are chirping when a rooster is crowing.",
                "Chicks are chirping while a rooster is crowing.",
                "Seagulls squawk, then hens and chicks chirp and a rooster crows thrice as waves break against the shore.",
                "Waves breaking on a shore and seagulls squawking followed by hens and chicks chirping and a rooster crowing three times",
                "Many varieties of bird sing their songs, including a crowing cock.",
            ),
            (
                "A liquid is completely squeezed out of a tube.",
                "A liquid is squeezed out of a tube until it is empty.",
                "An air bladder being emptied into a jelly like material.",
                "Something is being squeezed out of a bottle with difficulty.",
                "The last of the liquid soap is being squeezed out of the bottle.",
            ),
        ]

    def get_raw(self, name: str, index: int) -> Any:
        if name == "audio":
            return torch.full((3,), index)
        elif name == "captions":
            return self._all_captions[index]
        else:
            raise ValueError(f"Invalid index {index=}.")

    def get(self, name: str, index: int) -> Any:
        return self.get_raw(name, index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "index": index,
            "audio": self.get("audio", index),
            "captions": self.get("captions", index),
        }

    def __len__(self) -> int:
        return len(self._all_captions)


class ZipDataset(Dataset):
    def __init__(
        self,
        *datasets: SizedDataset,
        transforms: Optional[dict[str, Optional[nn.Module]]] = None,
    ) -> None:
        if len(datasets) > 0 and any(
            len(dset) != len(datasets[0]) for dset in datasets
        ):
            raise ValueError("Invalid datasets lengths for ZipDatasets.")

        if transforms is None:
            transforms = {}

        super().__init__()
        self._datasets = datasets
        self._transforms = transforms

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {}
        for dset in self._datasets:
            item |= dset[index]

        for key, transform in self._transforms.items():
            if key not in item.keys():
                raise ValueError(
                    f"Invalid transform {key=} for {self.__class__.__name__}."
                )
            if transform is not None:
                item[key] = transform(item[key])
        return item

    def __len__(self) -> int:
        return min(map(len, self._datasets))


def filter_audio_sizes(
    dset: AACDataset,
    min_audio_size: float = 0.0,
    max_audio_size: float = math.inf,
    verbose: int = 0,
) -> list[int]:

    fpaths = [dset.get("fpath", i) for i in range(len(dset))]
    infos = [
        torchaudio.info(fpath)  # type: ignore
        for fpath in tqdm.tqdm(
            fpaths,
            desc=f"Loading infos for filter audio not in [{min_audio_size}, {max_audio_size}] seconds...",
            disable=verbose <= 0,
        )
    ]
    indexes = [
        i
        for i, info in enumerate(infos)
        if min_audio_size <= info.num_frames / info.sample_rate <= max_audio_size
    ]

    if len(indexes) < len(dset):
        if verbose >= 1:
            logging.info(
                f"Exclude {len(dset) - len(indexes)}/{len(dset)} files with audio size not in [{min_audio_size}, {max_audio_size}] seconds."
            )
    return indexes
