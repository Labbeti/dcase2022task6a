#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Provides modules to use SpecAugment
BASED ON https://github.com/qiuqiangkong/sound_event_detection_dcase2017_task4
MODIFIED: Yes (typing, spectrogram reshape, add probability of specaugment)
"""

import random

import torch

from torch import nn, Tensor


class DropStripes(nn.Module):
    def __init__(self, dim: int, drop_width: int, stripes_num: int) -> None:
        """Drop stripes.

        Args:
                dim: int, dimension along which to drop
                drop_width: int, maximum width of stripes to drop
                stripes_num: int, how many stripes to drop
        """
        # Note: dim 2: time; dim 3: frequency
        if dim not in (2, 3):
            raise ValueError(f"Invalid argument dim={self.dim}. (expected 2 or 3)")
        if drop_width <= 0:
            raise ValueError(
                f"Invalid argument {drop_width=} in {self.__class__.__name__}. (expected a value > 0)"
            )

        super().__init__()
        self.dim = dim
        self.drop_width = drop_width
        self.stripes_num = stripes_num

    def forward(self, input: Tensor) -> Tensor:
        """
        :param input: (batch_size, channels, time_steps, freq_bins) pr (channels, time_steps, freq_bins) tensor
        :return: Same shape as input.
        """
        if input.ndim == 3:
            # found (channels, time_steps, freq_bins)
            input = input.unsqueeze_(dim=0)
            no_batch = True
        else:
            no_batch = False

        assert input.ndim == 4, f"found {input.ndim=}, but expected 4"

        if self.training:
            batch_size = input.shape[0]
            total_width = input.shape[self.dim]

            for n in range(batch_size):
                self.transform_slice(input[n], total_width)

        if no_batch:
            input = input.squeeze_(0)

        return input

    def transform_slice(self, inp: Tensor, total_width: int) -> None:
        """inp: (channels, time_steps, freq_bins)"""
        # Add: If audio is empty, do nothing
        if total_width == 0:
            return
        # Add: If audio is shorter than self.drop_width, clip drop width.
        drop_width = min(self.drop_width, total_width)

        for _ in range(self.stripes_num):
            distance = torch.randint(low=0, high=drop_width, size=())
            bgn = torch.randint(low=0, high=total_width - distance, size=())

            if self.dim == 2:
                inp[:, bgn : bgn + distance, :] = 0
            elif self.dim == 3:
                inp[:, :, bgn : bgn + distance] = 0
            else:
                raise ValueError(f"Invalid argument dim={self.dim}. (expected 2 or 3)")


class SpecAugmentation(nn.Module):
    def __init__(
        self,
        time_drop_width: int,
        time_stripes_num: int,
        freq_drop_width: int,
        freq_stripes_num: int,
        p: float = 1.0,
    ) -> None:
        """Spec augmentation.
        [ref] Park, D.S., Chan, W., Zhang, Y., Chiu, C.C., Zoph, B., Cubuk, E.D.
        and Le, Q.V., 2019. Specaugment: A simple data augmentation method
        for automatic speech recognition. arXiv preprint arXiv:1904.08779.

        Args:
            time_drop_width: int
            time_stripes_num: int
            freq_drop_width: int
            freq_stripes_num: int
        """
        super().__init__()
        self._p = p

        self.time_dropper = DropStripes(
            dim=2, drop_width=time_drop_width, stripes_num=time_stripes_num
        )
        self.freq_dropper = DropStripes(
            dim=3, drop_width=freq_drop_width, stripes_num=freq_stripes_num
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.training and (self._p >= 1.0 or random.random() < self._p):
            return self._forward_impl(x)
        else:
            return x

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.time_dropper(x)
        x = self.freq_dropper(x)
        return x
