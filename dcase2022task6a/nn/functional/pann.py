#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from torch import nn


def init_layer(layer) -> None:
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn) -> None:
    """Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


def do_mixup(x, mixup_lambda):
    """Mixup x of even indexes (0, 2, 4, ...) with x of odd indexes
    (1, 3, 5, ...).

    Args:
      x: (batch_size * 2, ...)
      mixup_lambda: (batch_size * 2,)

    Returns:
      out: (batch_size, ...)
    """
    out = (
        x[0::2].transpose(0, -1) * mixup_lambda[0::2]
        + x[1::2].transpose(0, -1) * mixup_lambda[1::2]
    ).transpose(0, -1)
    return out


def interpolate(x, ratio):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.

    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate

    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output, frames_num):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.

    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad

    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    pad = framewise_output[:, -1:, :].repeat(
        1, frames_num - framewise_output.shape[1], 1
    )
    """tensor for padding"""

    output = torch.cat((framewise_output, pad), dim=1)
    """(batch_size, frames_num, classes_num)"""

    return output
