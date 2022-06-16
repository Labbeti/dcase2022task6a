#!/usr/bin/env python
# -*- coding: utf-8 -*-

import inspect
import tqdm

from typing import Any, Callable, Iterable, Optional, Union

import torch

from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from dcase2022task6a.nn.functional.misc import (
    pad_sequence_rec,
    pad_sequence_1d,
    gumbel_softmax,
    gumbel_log_softmax,
)


class Mean(nn.Module):
    def __init__(self, dim: Optional[int]) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        if self.dim is None:
            return x.mean()
        else:
            return x.mean(dim=self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int) -> None:
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(self.dim0, self.dim1)

    def extra_repr(self) -> str:
        return f"{self.dim0}, {self.dim1}"


class Squeeze(nn.Module):
    def __init__(self, dim: Optional[int] = None, inplace: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if not self.inplace:
            if self.dim is None:
                return x.squeeze()
            else:
                return x.squeeze(self.dim)
        else:
            if self.dim is None:
                return x.squeeze_()
            else:
                return x.squeeze_(self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, inplace={self.inplace}"


class Unsqueeze(nn.Module):
    def __init__(self, dim: int, inplace: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if not self.inplace:
            return x.unsqueeze(self.dim)
        else:
            return x.unsqueeze_(self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, inplace={self.inplace}"


class AmplitudeToLog(nn.Module):
    def __init__(self, eps: float = torch.finfo(torch.float).eps) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, data: Tensor) -> Tensor:
        return torch.log(data + self.eps)


class Normalize(nn.Module):
    def forward(self, data: Tensor) -> Tensor:
        return F.normalize(data)


class Lambda(nn.Module):
    def __init__(self, fn: Callable, **default_kwargs) -> None:
        """Wrap a callable function or object to a Module."""
        super().__init__()
        self.fn = fn
        self.default_kwargs = default_kwargs

    def forward(self, *args, **kwargs) -> Any:
        kwargs = self.default_kwargs | kwargs
        return self.fn(*args, **kwargs)

    def extra_repr(self) -> str:
        if inspect.isfunction(self.fn):
            return self.fn.__name__
        elif inspect.ismethod(self.fn):
            return self.fn.__qualname__
        else:
            return self.fn.__class__.__name__


class Reshape(nn.Module):
    def __init__(self, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return torch.reshape(x, self.shape)


class Print(nn.Module):
    def __init__(
        self,
        preprocess: Optional[Callable] = None,
        prefix: str = "DEBUG - ",
    ) -> None:
        super().__init__()
        self._preprocess = preprocess
        self._prefix = prefix

    def forward(self, x: Any) -> Any:
        x_out = x
        if self._preprocess is not None:
            x = self._preprocess(x)
        print(f"{self._prefix}{x=}")
        return x_out


class ToTensor(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs

    def forward(self, inp: list, *args, **kwargs) -> Tensor:
        kwargs = self.kwargs | kwargs
        return torch.as_tensor(inp, *args, **kwargs)


class Duplicate(nn.Module):
    def __init__(
        self, n: int, post_transforms: Optional[list[nn.Module]] = None
    ) -> None:
        if post_transforms is not None and len(post_transforms) != n:
            raise ValueError(
                f"Invalid post_transforms length {len(post_transforms)} with {n=}."
            )
        super().__init__()
        self._n = n
        self._post_transforms = post_transforms

    def forward(self, inp: Any) -> list[Any]:
        if self._post_transforms is None:
            return [inp for _ in range(self._n)]
        else:
            return [transform(inp) for transform in self._post_transforms]


class TensorTo(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        kwargs = self.kwargs | kwargs
        return x.to(**kwargs)


class Permute(nn.Module):
    def __init__(self, *args) -> None:
        super().__init__()
        self._dims = tuple(args)

    def forward(self, x: Tensor) -> Tensor:
        return x.permute(self._dims)

    def extra_repr(self) -> str:
        return ", ".join(map(str, self._dims))


class Div(nn.Module):
    def __init__(
        self,
        divisor: Union[float, Tensor],
        rounding_mode: str = "trunc",
    ) -> None:
        super().__init__()
        self.divisor = divisor
        self.rounding_mode = rounding_mode

    def forward(self, x: Tensor) -> Tensor:
        return x.div(self.divisor, rounding_mode=self.rounding_mode)


class PadSequence(nn.Module):
    def __init__(self, pad_value: float, batch_first: bool = True) -> None:
        super().__init__()
        self._pad_value = pad_value
        self._batch_first = batch_first

    def forward(self, sequence: list[Tensor]) -> Tensor:
        return pad_sequence(sequence, self._batch_first, self._pad_value)


class PadSequenceRec(nn.Module):
    def __init__(self, pad_value: float) -> None:
        super().__init__()
        self._pad_value = pad_value

    def forward(self, sequence: list[Tensor]) -> Tensor:
        return pad_sequence_rec(sequence, self._pad_value)


class PadSequence1D(nn.Module):
    def __init__(self, pad_value: float) -> None:
        super().__init__()
        self._pad_value = pad_value

    def forward(self, sequence: list[Tensor]) -> Tensor:
        return pad_sequence_1d(sequence, self._pad_value)


class ParallelDict(nn.ModuleDict):
    """Compute output of each submodule value when forward(.) is called."""

    def __init__(
        self, modules: Optional[dict[str, nn.Module]] = None, verbose: bool = False
    ) -> None:
        super().__init__(modules)
        self._verbose = verbose

    def forward(self, *args, **kwargs) -> dict[str, Any]:
        tqdm_obj = tqdm.tqdm(
            self.items(), desc="{self.__class__.__name__}", disable=not self._verbose
        )
        outs = {}
        for name, module in tqdm_obj:
            tqdm_obj.set_description(
                f"{self.__class__.__name__}:{module.__class__.__name__}"
            )
            outs[name] = module(*args, **kwargs)
        return outs


class ParallelList(nn.ModuleList):
    def __init__(
        self, modules: Optional[Iterable[nn.Module]] = (), verbose: bool = False
    ) -> None:
        super().__init__(modules)
        self._verbose = verbose

    def forward(self, *args, **kwargs) -> list[Any]:
        tqdm_obj = tqdm.tqdm(
            self,
            disable=not self._verbose,
            desc=f"{self.__class__.__name__}",
        )
        outs = []
        for module in tqdm_obj:
            tqdm_obj.set_description(
                f"{self.__class__.__name__}:{module.__class__.__name__}"
            )
            outs.append(module(*args, **kwargs))
        return outs


class SequentialDict(nn.Sequential):
    def forward(self, **kwargs) -> dict[str, Any]:
        x = kwargs
        for module in self:
            x = module(**x)
            if not isinstance(x, dict):
                raise TypeError(
                    f"Invalid output type {type(x)} for {self.__class__.__name__}."
                )
        return x


class GumbelSoftmax(nn.Module):
    def __init__(
        self,
        tau: float = 1.0,
        hard: bool = False,
        dim: int = -1,
        generator: Union[None, torch.Generator] = None,
    ) -> None:
        super().__init__()
        self.tau = tau
        self.hard = hard
        self.dim = dim
        self.generator = generator

    def forward(self, logits: Tensor) -> Tensor:
        return gumbel_softmax(logits, self.tau, self.hard, self.dim, self.generator)


class GumbelLogSoftmax(nn.Module):
    def __init__(
        self,
        tau: float = 1.0,
        hard: bool = False,
        dim: int = -1,
        generator: Union[None, torch.Generator] = None,
    ) -> None:
        super().__init__()
        self.tau = tau
        self.hard = hard
        self.dim = dim
        self.generator = generator

    def forward(self, logits: Tensor) -> Tensor:
        return gumbel_log_softmax(logits, self.tau, self.hard, self.dim, self.generator)


class Standardize(nn.Module):
    def __init__(self, unbiased_std: bool = True) -> None:
        super().__init__()
        self.unbiased_std = unbiased_std

    def forward(self, x: Tensor) -> Tensor:
        x = (x - x.mean()) / x.std(unbiased=self.unbiased_std)
        return x
