#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import logging
import os.path as osp
import tqdm

from typing import Any, Optional, Sequence, Sized, Union

import h5py
import numpy as np
import torch

from torch import nn, Tensor
from torch.utils.data.dataset import Dataset


HDF_ENCODING = "utf-8"
# Type for strings
HDF_STRING_DTYPE = h5py.string_dtype(HDF_ENCODING, None)
# Type for empty lists
HDF_VOID_DTYPE = h5py.opaque_dtype("V1")
SHAPE_SUFFIX = "_shape"


def _all_eq(seq: Sequence[Any]) -> bool:
    """Returns True if all element in list are the same."""
    if len(seq) == 0:
        return True
    else:
        first = seq[0]
        return all(first == elt for elt in seq[1:])


def _get_shape_and_dtype(
    value: Union[int, float, str, Tensor, list]
) -> tuple[tuple, str]:
    """Returns the shape and the hdf_dtype for an input."""
    if isinstance(value, int):
        shape = ()
        hdf_dtype = "i"

    elif isinstance(value, float):
        shape = ()
        hdf_dtype = "f"

    elif isinstance(value, str):
        shape = ()
        hdf_dtype = HDF_STRING_DTYPE

    elif isinstance(value, Tensor):
        shape = tuple(value.shape)
        if value.is_floating_point():
            hdf_dtype = "f"
        else:
            hdf_dtype = "i"

    elif isinstance(value, (list, tuple)):
        if len(value) == 0:
            shape = (0,)
            hdf_dtype = HDF_VOID_DTYPE
        else:
            sub_shapes_and_dtypes = list(map(_get_shape_and_dtype, value))
            sub_shapes = [shape for shape, _ in sub_shapes_and_dtypes]
            sub_dtypes = [dtype for _, dtype in sub_shapes_and_dtypes]
            sub_dims = list(map(len, sub_shapes))

            if not _all_eq(sub_dims):
                raise TypeError(
                    f"Unsupported list of heterogeneous shapes lengths. (found {sub_dims=})"
                )
            if not _all_eq(sub_dtypes):
                raise TypeError(
                    f"Unsupported list of heterogeneous types. (found {sub_dtypes=})"
                )
            # Check for avoid ragged array like [["a", "b"], ["c"], ["d", "e"]]
            if not _all_eq(sub_shapes):
                raise TypeError(
                    f"Unsupported list of heterogeneous shapes. (found {sub_shapes=} for {value=})"
                )

            max_subshape = tuple(
                max(shape[i] for shape in sub_shapes) for i in range(len(sub_shapes[0]))
            )
            shape = (len(value),) + max_subshape
            hdf_dtype = sub_dtypes[0]
    else:
        raise TypeError(
            f"Unsupported type {value.__class__.__name__}[{value[0].__class__.__name__}] in function get_shape_and_dtype."
        )

    return shape, hdf_dtype


def _decode_rec(value: Union[bytes, list], encoding: str) -> Union[str, list]:
    """Decode bytes to str with the specified encoding. Works recursively on list of str, list of list of str, etc."""
    if isinstance(value, bytes):
        return value.decode(encoding=encoding)
    else:
        return [_decode_rec(elt, encoding) for elt in value]


def pack_to_hdf(
    dataset: Any,
    hdf_fpath: str,
    pre_save_transforms: Optional[dict[str, Optional[nn.Module]]] = None,
    post_save_transforms: Optional[dict[str, Optional[nn.Module]]] = None,
    overwrite: bool = False,
    metadata: str = "",
    verbose: int = 0,
) -> "HDFDataset":
    """
    Pack a dataset to HDF file.

    :param dataset: The sized dataset to pack. Must be sized and all items must be of dict type.
        The key of each dictionaries are strings and values can be int, float, str, Tensor, non-empty list[int], non-empty list[float], non-empty list[str].
        If values are tensors or lists, the number of dimensions must be the same for all items in the dataset.
    :param hdf_fpath: The path to the HDF file.
    :param pre_save_transform: The optional transform to apply to audio returned by the dataset BEFORE storing it in HDF file.
        Can be used for deterministic transforms like Resample, LogMelSpectrogram, etc. defaults to None.
    :param post_save_transform: The optional transform to apply to audio returned by the HDF file.
        Can be used for non-deterministic transforms like augmentations, but it will not be pre-computed. defaults to None.
    :param overwrite: If True, the file hdf_fpath can be overwritten. defaults to False.
    :param metadata: Additional metadata string to add to the hdf file. defaults to ''.
    :param verbose: Verbose level. defaults to 0.
    :returns: The HDFDataset object created and opened.
    """
    # Check inputs
    if not isinstance(dataset, Dataset):
        raise TypeError(
            f"Cannot pack a non-dataset '{dataset.__class__.__name__}'. (found {isinstance(dataset, Dataset)=})"
        )
    if not isinstance(dataset, Sized):
        raise TypeError(
            f"Cannot pack a non-sized dataset '{dataset.__class__.__name__}'. (found {isinstance(dataset, Sized)=})"
        )
    if osp.exists(hdf_fpath) and not osp.isfile(hdf_fpath):
        raise RuntimeError(f"Item {hdf_fpath=} exists but it is not a file.")
    if not overwrite and osp.isfile(hdf_fpath):
        raise ValueError(
            f"Cannot overwrite file {hdf_fpath}. Please remove it or use overwrite=True option."
        )

    if pre_save_transforms is None:
        pre_save_transforms = {}

    # Step 1: Init max_shapes and hdf_dtypes with the first item
    item_i = dataset[0]
    if not isinstance(item_i, dict):
        raise ValueError(
            f"Invalid item type for {dataset.__class__.__name__}. (expected dict but found {type(item_i)})"
        )

    max_shapes = {}
    hdf_dtypes = {}
    for name, value in item_i.items():
        transform = pre_save_transforms.get(name, None)
        if transform is not None:
            value = transform(value)
        shape, hdf_dtype = _get_shape_and_dtype(value)
        max_shapes[name] = shape
        hdf_dtypes[name] = hdf_dtype

    # Compute max_shapes with a first pass through the whole dataset
    for i in tqdm.trange(
        1,
        len(dataset),
        desc="Step 1/2: Computing max_shapes...",
        disable=verbose <= 0,
    ):
        item_i = dataset[i]
        if not isinstance(item_i, dict):
            raise ValueError(
                f"Invalid item type for {dataset.__class__.__name__}. (expected dict, found {type(item_i)})"
            )
        if set(item_i.keys()) != set(max_shapes.keys()):
            raise ValueError(
                f"Invalid item keys. Every item must return a dict containing the same keys. (found {item_i.keys()=} != {max_shapes.keys()=} with {i=}/{len(dataset)})"
            )

        for name, value in item_i.items():
            transform = pre_save_transforms.get(name, None)
            if transform is not None:
                value = transform(value)
            shape, hdf_dtype = _get_shape_and_dtype(value)
            if hdf_dtype != hdf_dtypes[name]:
                raise ValueError(
                    f"Found a differents hdf_dtypes in the dataset. (found {hdf_dtype=} != {hdf_dtypes[name]})"
                )
            if len(shape) != len(max_shapes[name]):
                raise ValueError(
                    f"Found another value with different number of dims for {dataset.__class__.__name__}[{i}][{name}]."
                    f"(found {len(max_shapes[name])=} != {len(shape)=})"
                )
            if hdf_dtype == HDF_STRING_DTYPE:
                try:
                    np.array(value, dtype=str)
                except ValueError:
                    raise ValueError(
                        f"Unsupported ragged array for {name=}. (found {value=} at {i=})"
                    )
            max_shapes[name] = tuple(map(max, zip(max_shapes[name], shape)))

    if set(max_shapes.keys()) != set(hdf_dtypes.keys()):
        raise RuntimeError(
            f"Internal error: found differents keys in max_shapes and hdf_dtypes. (found {max_shapes.keys()=} != {hdf_dtypes.keys()=})"
        )

    if verbose >= 2:
        logging.debug(f"Found max_shapes:\n{max_shapes}")
        logging.debug(f"Found hdf_dtypes:\n{hdf_dtypes}")

    now = datetime.datetime.now()
    attributes = {
        "creation_date": now.strftime("%Y-%m-%d_%H-%M-%S"),
        "source_dataset": dataset.__class__.__name__,
        "length": len(dataset),
        "metadata": str(metadata),
        "author": "Etienne Labbé (Labbeti)",
        "author_mail": "etienne _dot_ labbe31 _at_ gmail _dot_ com",
        "encoding": HDF_ENCODING,
    }

    with h5py.File(hdf_fpath, "w") as hdf_file:
        # Step 2: Build hdf datasets in file
        hdf_dsets = {}
        for name, max_shape in max_shapes.items():
            hdf_dtype = hdf_dtypes.get(name)

            kwargs = {}
            if hdf_dtype == "i":
                kwargs["fillvalue"] = 0
            elif hdf_dtype == "f":
                kwargs["fillvalue"] = 0.0
            elif hdf_dtype in (HDF_STRING_DTYPE, HDF_VOID_DTYPE):
                pass
            else:
                raise ValueError(
                    f"Unknown value {hdf_dtype=}. (with {name=} and {name in hdf_dtypes=})"
                )

            hdf_dsets[name] = hdf_file.create_dataset(
                name, (len(dataset),) + max_shape, hdf_dtype, **kwargs
            )

            if max_shape != ():
                shape_name = f"{name}{SHAPE_SUFFIX}"
                hdf_dsets[shape_name] = hdf_file.create_dataset(
                    shape_name, (len(dataset), len(max_shape)), "i", fillvalue=-1
                )

        # Fill hdf datasets with a second pass through the whole dataset
        for i in tqdm.trange(
            len(dataset),
            desc="Step 2/2: Pack data into HDF...",
            disable=verbose <= 0,
        ):
            item_i = dataset[i]

            for name, value in item_i.items():
                transform = pre_save_transforms.get(name, None)
                if transform is not None:
                    value = transform(value)
                shape, _hdf_dtype = _get_shape_and_dtype(value)

                # Check every shape
                max_shape = max_shapes[name]
                if len(shape) != len(max_shape):
                    raise ValueError(
                        f"Invalid number of dimension in audio (expected {len(max_shape)}, found {len(shape)})."
                    )
                if any(
                    shape_i > max_shape_i
                    for shape_i, max_shape_i in zip(shape, max_shape)
                ):
                    raise ValueError(
                        f"At least 1 dim of audio shape is above the maximal value (found {shape=} and {max_shape=})."
                    )

                if isinstance(value, Tensor) and value.is_cuda:
                    value = value.cpu()

                # Note: "dset_audios[slices]" is a generic version of "dset_audios[i, :shape_0, :shape_1]"
                slices = (i,) + tuple(slice(shape_i) for shape_i in shape)
                try:
                    hdf_dsets[name][slices] = value
                except TypeError as err:
                    logging.error(
                        f"Cannot set data {value} into {hdf_dsets[name][slices].shape} ({name=}, {i=})"
                    )
                    raise err

                # Store original shape if needed
                shape_name = f"{name}{SHAPE_SUFFIX}"
                if shape_name in hdf_dsets.keys():
                    hdf_dsets[shape_name][i] = shape

        for name, attribute in attributes.items():
            hdf_file.attrs[name] = attribute

    return HDFDataset(hdf_fpath, transforms=post_save_transforms)


class HDFDataset(Dataset):
    def __init__(
        self,
        hdf_fpath: str,
        transforms: Optional[dict[str, Optional[nn.Module]]] = None,
        keep_padding: Sequence[str] = (),
        include_keys: Optional[Sequence[str]] = None,
        exclude_keys: Sequence[str] = (),
        open_hdf: bool = True,
    ) -> None:
        """
        :param hdf_fpath: The path to the HDF file.
        :param transforms: The transform to apply values (Tensor). default to None.
                Keys can be any key available in the HDF file.
        :param keep_padding: Keys for keep padding values. defaults to ().
        :param include_keys: If not None, this list will defines which values will be used as output of getitem.
        :param exclude_keys: Exclude keys from getitem. defaults to ().
        :param open_hdf: If True, open the HDF file at start. defaults to True.
        """
        if not osp.isfile(hdf_fpath):
            raise FileNotFoundError(f"Cannot find HDF file in path {hdf_fpath=}.")

        if transforms is None:
            transforms = {}
        transforms = {
            name: transform
            for name, transform in transforms.items()
            if transform is not None
        }

        super().__init__()
        self._hdf_fpath = hdf_fpath
        self._transforms = transforms
        self._keep_padding = keep_padding
        self._include_keys = include_keys
        self._exclude_keys = exclude_keys

        self._hdf_file: Any = None

        if open_hdf:
            self.open()

    def __del__(self) -> None:
        if self.is_open():
            self.close()

    def __exit__(self) -> None:
        if self.is_open():
            self.close()

    def __hash__(self) -> int:
        hash_ = (
            hash(self._hdf_fpath)
            + sum(map(hash, self._transforms.keys()))
            + sum(map(hash, self._keep_padding))
            + sum(map(hash, self._exclude_keys))
        )
        return hash_

    def __len__(self) -> int:
        return self._hdf_file.attrs["length"]

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.get_dict_item(index)

    def get_dict_item(self, index: int) -> dict[str, Any]:
        dict_item = {key: self.get(key, index) for key in self.get_item_keys()}
        return dict_item

    def get_tuple_item(self, index: int) -> tuple:
        tuple_item = tuple(self.get(key, index) for key in self.get_item_keys())
        return tuple_item

    def get_raw(self, name: str, index: int) -> Any:
        if not self.is_open():
            raise RuntimeError(
                f"Cannot get_raw value with closed HDF file. ({self._hdf_file is not None=} and {bool(self._hdf_file)=})"
            )
        if name not in self._hdf_file.keys():
            raise ValueError(f"Unknown {name=}.")
        if not isinstance(index, int) or not (0 <= index < len(self)):
            raise ValueError(
                f"Invalid argument {index=}. (expected int in range [0,{len(self)}[)"
            )

        dset = self._hdf_file[name]
        value: Any = dset[index]

        # Remove the padding part
        shape_name = f"{name}{SHAPE_SUFFIX}"
        if shape_name in self._hdf_file.keys() and name not in self._keep_padding:
            shape = self._hdf_file[shape_name][index]
            slices = tuple(slice(shape_i) for shape_i in shape)
            value = value[slices]

        # Decode all bytes to string
        if dset.dtype == HDF_STRING_DTYPE:
            value = _decode_rec(value, HDF_ENCODING)
        # Convert numpy.array to torch.Tensor
        elif isinstance(value, np.ndarray):
            if dset.dtype != HDF_VOID_DTYPE:
                value = torch.from_numpy(value)
            else:
                value = value.tolist()
        # Convert numpy scalars
        elif np.isscalar(value) and hasattr(value, "item"):
            value = value.item()

        return value

    def get(self, name: str, index: int) -> Any:
        value = self.get_raw(name, index)
        transform = self._transforms.get(name, None)
        if transform is not None:
            value = transform(value)
        return value

    def get_attrs(self) -> dict[str, Any]:
        return self._hdf_file.attrs

    def get_max_shape(self, name: str) -> tuple[int, ...]:
        if not self.is_open():
            raise RuntimeError(
                f"Cannot get max_shape with a closed HDF file. ({self._hdf_file is not None=} and {bool(self._hdf_file)=})"
            )
        shape_name = f"{name}{SHAPE_SUFFIX}"
        return tuple(self._hdf_file[shape_name].shape[1:])

    def get_hdf_keys(self) -> tuple[str, ...]:
        if self.is_open():
            return tuple(self._hdf_file.keys())
        else:
            raise RuntimeError("Cannot get keys from a closed HDF file.")

    def get_item_keys(self) -> tuple[str, ...]:
        return tuple(
            key
            for key in self.get_hdf_keys()
            if (key not in self._exclude_keys)
            and (self._include_keys is None or key in self._include_keys)
        )

    def get_hdf_fpath(self) -> str:
        return self._hdf_fpath

    def open(self) -> None:
        if self.is_open():
            raise RuntimeError("Cannot open the HDF file twice.")
        self._hdf_file = h5py.File(self._hdf_fpath, "r")
        self._sanity_check()

    def close(self) -> None:
        if not self.is_open():
            raise RuntimeError("Cannot close the HDF file twice.")
        self._hdf_file.close()
        self._hdf_file = None

    def is_open(self) -> bool:
        return self._hdf_file is not None and bool(self._hdf_file)

    def _sanity_check(self) -> None:
        lens = [dset.shape[0] for dset in self._hdf_file.values()]
        if not _all_eq(lens) or lens[0] != len(self):
            logging.error(
                f"Incorrect length stored in HDF file. (found {lens=} and {len(self)=})"
            )

        for key in self._transforms.keys():
            if key not in self.get_hdf_keys():
                logging.error(
                    f"Found invalid transform {key=}. (expected one of hdf keys {self.get_hdf_keys()})"
                )

    def __repr__(self) -> str:
        return f"HDFDataset(hdf_fname={osp.basename(self._hdf_fpath)})"

    def __getstate__(self) -> dict[str, Any]:
        return {
            "hdf_fpath": self._hdf_fpath,
            "transforms": self._transforms,
            "keep_padding": self._keep_padding,
            "include_keys": self._include_keys,
            "exclude_keys": self._exclude_keys,
        }

    def __setstate__(self, data: dict[str, Any]) -> None:
        file_is_different = self._hdf_fpath != data["hdf_fpath"]
        is_open = self.is_open()

        self._hdf_fpath = data["hdf_fpath"]
        self._transforms = data["transforms"]
        self._keep_padding = data["keep_padding"]
        self._include_keys = data["include_keys"]
        self._exclude_keys = data["exclude_keys"]

        if file_is_different and is_open:
            self.close()  # close old hdf file
            self.open()  # open new hdf file
