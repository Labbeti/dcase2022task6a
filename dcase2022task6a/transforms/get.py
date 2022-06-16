#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path as osp
import pickle

from pathlib import Path
from typing import Any, Optional, Union

import torch
import yaml

from nnAudio.features import Gammatonegram
from torch import nn, Tensor
from torchaudio.transforms import Resample
from torchlibrosa.stft import Spectrogram, LogmelFilterBank

from dcase2022task6a.nn.encoders.passt import PASST
from dcase2022task6a.nn.modules.misc import (
    Lambda,
    Mean,
    Permute,
    Squeeze,
    Standardize,
    TensorTo,
    Unsqueeze,
)
from dcase2022task6a.transforms.audio.spec_aug import SpecAugmentation


def get_none() -> None:
    # Returns None. Can be used for hydra instantiations.
    return None


def get_pickle(
    fpath: Union[str, Path],
) -> nn.Module:
    if not isinstance(fpath, (str, Path)):
        raise TypeError(f"Invalid transform with pickle {fpath=}. (not a str or Path)")
    if not osp.isfile(fpath):
        raise FileNotFoundError(f"Invalid transform with pickle {fpath=}. (not a file)")

    with open(fpath, "rb") as file:
        transform = pickle.load(file)
    return transform


def get_resample_mean(
    src_sr: int,
    tgt_sr: int,
    mean_dim: Optional[int] = 0,
) -> nn.Module:
    if not isinstance(src_sr, int):
        raise ValueError(_get_error_message(src_sr))

    return nn.Sequential(
        Resample(src_sr, tgt_sr),
        Mean(dim=mean_dim),
    )


def get_resample_mean_passt(
    src_sr: int,
    tgt_sr: int,
    mean_dim: Optional[int] = 0,
    device: Union[None, str, torch.device] = "cpu",
) -> nn.Module:
    if not isinstance(src_sr, int):
        raise ValueError(_get_error_message(src_sr))

    if isinstance(device, str):
        device = torch.device(device)
    passt = PASST(freeze_mode="all").to(device)

    def get_passt_embs(audio: Tensor) -> Tensor:
        return passt(audio, return_dict=False, return_clip_embs=False)

    return nn.Sequential(
        Resample(src_sr, tgt_sr),
        Mean(dim=mean_dim),
        Unsqueeze(dim=0),
        TensorTo(device=device),
        Lambda(get_passt_embs),
    )


def get_resample_squeeze(
    src_sr: int,
    tgt_sr: int,
    squeeze_dim: Optional[int] = 0,
) -> nn.Module:
    if not isinstance(src_sr, int):
        raise ValueError(_get_error_message(src_sr))

    return nn.Sequential(
        Resample(src_sr, tgt_sr),
        Squeeze(dim=squeeze_dim),
    )


def get_resample_spectro_mean_spec_aug(
    src_sr: int,
    tgt_sr: int,
    window_size: int = 1024,
    hop_size: int = 320,
    mel_bins: int = 64,
    fmin: int = 50,
    fmax: int = 14000,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "reflect",
    ref: float = 1.0,
    amin: float = 1e-10,
    top_db: Optional[float] = None,
    freeze_parameters: bool = True,
    mean_dim: Optional[int] = 0,
    time_drop_width: int = 64,
    time_stripes_num: int = 2,
    freq_drop_width: int = 2,
    freq_stripes_num: int = 1,
    spec_aug_p: float = 1.0,
) -> nn.Module:
    if not isinstance(src_sr, int):
        raise ValueError(_get_error_message(src_sr))

    return nn.Sequential(
        Resample(src_sr, tgt_sr),
        Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=freeze_parameters,
        ),
        LogmelFilterBank(
            sr=tgt_sr,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=freeze_parameters,
        ),
        Mean(dim=mean_dim),
        SpecAugmentation(
            time_drop_width=time_drop_width,
            time_stripes_num=time_stripes_num,
            freq_drop_width=freq_drop_width,
            freq_stripes_num=freq_stripes_num,
            p=spec_aug_p,
        ),
    )


def get_resample_spectro_mean(
    src_sr: int,
    tgt_sr: int,
    window_size: int = 1024,
    hop_size: int = 320,
    mel_bins: int = 64,
    fmin: int = 50,
    fmax: int = 14000,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "reflect",
    ref: float = 1.0,
    amin: float = 1e-10,
    top_db: Optional[float] = None,
    freeze_parameters: bool = True,
    mean_dim: Optional[int] = 0,
) -> nn.Module:
    if not isinstance(src_sr, int):
        raise ValueError(_get_error_message(src_sr))

    return nn.Sequential(
        Resample(src_sr, tgt_sr),
        Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=freeze_parameters,
        ),
        LogmelFilterBank(
            sr=tgt_sr,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=freeze_parameters,
        ),
        Mean(dim=mean_dim),
    )


def get_resample_spectro_squeeze(
    src_sr: int,
    tgt_sr: int,
    window_size: int = 1024,
    hop_size: int = 320,
    mel_bins: int = 64,
    fmin: int = 50,
    fmax: int = 14000,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "reflect",
    ref: float = 1.0,
    amin: float = 1e-10,
    top_db: Optional[float] = None,
    freeze_parameters: bool = True,
    squeeze_dim: Optional[int] = 0,
) -> nn.Module:
    if not isinstance(src_sr, int):
        raise ValueError(_get_error_message(src_sr))

    return nn.Sequential(
        Resample(src_sr, tgt_sr),
        Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=freeze_parameters,
        ),
        LogmelFilterBank(
            sr=tgt_sr,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=freeze_parameters,
        ),
        Squeeze(dim=squeeze_dim),
    )


def get_resample_mean_gamma_perm(
    src_sr: int,
    tgt_sr: int,
    mean_dim: int = 0,
    n_fft: int = 1024,
    n_bins: int = 64,
    hop_length: int = 512,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "reflect",
    power: float = 2.0,
    htk: bool = False,
    fmin: Optional[float] = 20.0,
    fmax: Optional[float] = None,
    norm: float = 1,
    trainable_bins: bool = False,
    trainable_STFT: bool = False,
) -> nn.Module:
    if not isinstance(src_sr, int):
        raise ValueError(_get_error_message(src_sr))

    return nn.Sequential(
        Resample(src_sr, tgt_sr),
        Mean(dim=mean_dim),
        Gammatonegram(
            sr=tgt_sr,
            n_fft=n_fft,
            n_bins=n_bins,
            hop_length=hop_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
            power=power,
            htk=htk,
            fmin=fmin,
            fmax=fmax,
            norm=norm,
            trainable_bins=trainable_bins,
            trainable_STFT=trainable_STFT,
            verbose=False,
        ),
        Permute(0, 2, 1),
    )


def get_stand_resample_spectro_mean_spec_aug(
    src_sr: int,
    tgt_sr: int,
    window_size: int = 1024,
    hop_size: int = 320,
    mel_bins: int = 64,
    fmin: int = 50,
    fmax: int = 14000,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "reflect",
    ref: float = 1.0,
    amin: float = 1e-10,
    top_db: Optional[float] = None,
    freeze_parameters: bool = True,
    mean_dim: Optional[int] = 0,
    time_drop_width: int = 64,
    time_stripes_num: int = 2,
    freq_drop_width: int = 2,
    freq_stripes_num: int = 1,
    spec_aug_p: float = 1.0,
) -> nn.Module:
    if not isinstance(src_sr, int):
        raise ValueError(_get_error_message(src_sr))

    return nn.Sequential(
        Standardize(),
        Resample(src_sr, tgt_sr),
        Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=freeze_parameters,
        ),
        LogmelFilterBank(
            sr=tgt_sr,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=freeze_parameters,
        ),
        Mean(dim=mean_dim),
        SpecAugmentation(
            time_drop_width=time_drop_width,
            time_stripes_num=time_stripes_num,
            freq_drop_width=freq_drop_width,
            freq_stripes_num=freq_stripes_num,
            p=spec_aug_p,
        ),
    )


def get_stand_resample_spectro_mean(
    src_sr: int,
    tgt_sr: int,
    window_size: int = 1024,
    hop_size: int = 320,
    mel_bins: int = 64,
    fmin: int = 50,
    fmax: int = 14000,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "reflect",
    ref: float = 1.0,
    amin: float = 1e-10,
    top_db: Optional[float] = None,
    freeze_parameters: bool = True,
    mean_dim: Optional[int] = 0,
) -> nn.Module:
    if not isinstance(src_sr, int):
        raise ValueError(_get_error_message(src_sr))

    return nn.Sequential(
        Standardize(),
        Resample(src_sr, tgt_sr),
        Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=freeze_parameters,
        ),
        LogmelFilterBank(
            sr=tgt_sr,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=freeze_parameters,
        ),
        Mean(dim=mean_dim),
    )


def _get_error_message(src_sr: Any) -> str:
    defaults_srs = {"clotho": 44100, "audiocaps": 32000, "macs": 48000}
    defaults_srs = yaml.dump(defaults_srs, sort_keys=False)
    message = (
        "\n"
        f"Invalid sr={src_sr} for get_resample_mean() function.\n"
        f"Please specify explicitely the source sample rate in Hz with data.sr=SAMPLE_RATE.\n"
        f"BE CAREFUL, sample rate can be different if you use pre-processed HDF files.\n"
        f"Defaults sample rates are:\n{defaults_srs}"
    )
    return message
