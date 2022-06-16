#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Callable, Generator, Iterable, Optional, Sequence, Sized, Union

import torch

from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


def sort_batch_by_lengths(
    batch: dict[str, Tensor],
    key: str = "audio_lens",
) -> dict[str, Tensor]:
    indexes = torch.argsort(batch["audio_lens"])

    keys = list(batch.keys())
    result = {}
    for key in keys:
        value = batch[key]
        del batch[key]
        if isinstance(value, Tensor):
            result[key] = value[indexes]
        elif isinstance(value, Sequence):
            result[key] = [value[i] for i in indexes]
        else:
            raise ValueError(f"Unsupported value type. ({value=})")
    return result


def randperm_diff(size: int) -> Tensor:
    """This function ensure that every value i cannot be the element at index i.

    Example 1
    ----------
    >>> torch.randperm(5)
    tensor([1, 4, 2, 5, 0])  # 2 is the element of index 2 !
    >>> randperm_diff(5)  # the function ensure that every value i cannot be the element at index i
    tensor([2, 0, 4, 1, 3])

    :param size: The number of indexes. Cannot be < 2.
    :returns: A tensor of shape (size,).
    """
    if size < 2:
        raise ValueError(f"Invalid argument {size=} < 2 for randperm_diff.")
    perm = torch.randperm(size)
    arange = torch.arange(size)
    while perm.eq(arange).any():
        perm = torch.randperm(size)
    return perm


def randperm_groupdiff(size: int, group_size: int) -> Tensor:
    """TODO

    :param size: The number of indexes. Cannot be < 2.
    :param group_size: TODO
    :returns: A tensor of shape (size,).
    """
    if size < 2:
        raise ValueError(f"Invalid argument {size=} < 2 for randperm_diff.")
    if size % group_size != 0:
        raise ValueError("Expected size divisible by group_size.")

    perm = torch.randperm(size)
    arange_group = torch.arange(size).div(group_size, rounding_mode="floor")
    while perm.div(group_size, rounding_mode="floor").eq(arange_group).any():
        perm = torch.randperm(size)
    return perm


def check_diff(perm: Tensor) -> bool:
    size = len(perm)
    arange = torch.arange(size)
    return bool(perm.eq(arange).any().item())


def check_groupdiff(indexes: Tensor, group_size: int) -> bool:
    size = len(indexes)
    arange_group = torch.arange(size).div(group_size, rounding_mode="floor")
    return bool(
        indexes.div(group_size, rounding_mode="floor").eq(arange_group).any().item()
    )


def randint_group(size: int, group_size: int) -> Tensor:
    assert size >= 2
    assert size % group_size == 0
    n_groups = size // group_size
    assert n_groups >= 2

    indexes = torch.empty(size, dtype=torch.long)
    for i in range(size):
        current_group = i // group_size
        other_group = torch.randint(n_groups - 1, ()).item()
        other_group = (
            other_group if other_group < current_group else (other_group + 1) % n_groups
        )
        local_group_index = torch.randint(group_size, ()).item()
        indexes[i] = other_group * group_size + local_group_index

    return indexes


def generate_square_subsequent_mask(
    size: int,
    device: Optional[torch.device] = None,
) -> Tensor:
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).

    NOTE : BASED ON PYTORCH IMPLEMENTATION in nn.Transformer

    :param size: The size of the output tensor.
    :param device: The device of the output tensor.
    :returns: A tensor of shape (size, size)

    Example 1
    ----------
    >>> generate_square_subsequent_mask(6)
    tensor([[0., -inf, -inf, -inf, -inf, -inf],
            [0., 0., -inf, -inf, -inf, -inf],
            [0., 0., 0., -inf, -inf, -inf],
            [0., 0., 0., 0., -inf, -inf],
            [0., 0., 0., 0., 0., -inf],
            [0., 0., 0., 0., 0., 0.]])
    """
    mask = (torch.triu(torch.ones(size, size, device=device)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def generate_shifted_sq_mask(
    size: int,
    right_shift: int = 0,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Example 1
    ----------
    >>> generate_shifted_sq_mask(6, 2)
    tensor([[0., 0., 0., -inf, -inf, -inf],
            [0., 0., 0., 0., -inf, -inf],
            [0., 0., 0., 0., 0., -inf],
            [0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.]])
    """
    mask = (
        torch.triu(torch.ones(size, size, device=device), diagonal=-right_shift) == 1
    ).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def count_params(model: nn.Module, only_trainable: bool = False) -> int:
    return sum(
        param.numel()
        for param in model.parameters()
        if not only_trainable or param.requires_grad
    )


def checksum(
    module_or_tensor: Union[nn.Module, Tensor],
    only_trainable: bool = False,
    type_: Optional[Callable] = int,
) -> Any:
    if isinstance(module_or_tensor, nn.Module):
        value = sum(
            param.nansum().item()
            for param in module_or_tensor.parameters()
            if not only_trainable or param.requires_grad
        )
    elif isinstance(module_or_tensor, Tensor):
        if not only_trainable or module_or_tensor.requires_grad:
            value = module_or_tensor.nansum().item()
        else:
            value = 0
    else:
        raise ValueError(
            f"Invalid argument type '{module_or_tensor.__class__.__name__}' with value '{module_or_tensor}'. (expected Module or Tensor)"
        )

    if type_ is not None:
        value = type_(value)
    return value


def module_eq(m1: nn.Module, m2: nn.Module) -> bool:
    n_params1 = sum(1 for _ in m1.parameters())
    n_params2 = sum(1 for _ in m2.parameters())
    return n_params1 == n_params2 and all(
        p1.shape == p2.shape and p1.eq(p2).all()
        for p1, p2 in zip(m1.parameters(), m2.parameters())
    )


def tensor_eq(t1: Tensor, t2: Tensor) -> bool:
    return t1.shape == t2.shape and bool(t1.eq(t2).all().item())


def tensor_close(t1: Tensor, t2: Tensor) -> bool:
    return t1.shape == t2.shape and torch.allclose(t1, t2)


def tensors_list_to_tensor(
    tensors: list[Tensor],
    pad_value: float,
    batch_first: bool = True,
) -> Tensor:
    """Pad a list of tensors to a tensor.

    :param tensors: List of N tensors with the same number of dims and with a different size at the first dim.
    :param pad_value: The value used for fill the tensors.
    :returns: (N, *)
    """
    return pad_sequence(tensors, batch_first=batch_first, padding_value=pad_value)


def tensors_list_to_lengths(tensors: list[Tensor], dim: int = -1) -> Tensor:
    """Return the size of the tensor at a specific dim.

    :param tensors: List of N tensors.
    :param dim: The dimension of the output sizes. defaults to -1.
    :returns: A tensor of size N with the sizes.
    """
    return torch.as_tensor([tensor.shape[dim] for tensor in tensors])


def tensor_to_tensors_list(
    tensor: Tensor,
    pad_value: Optional[float] = None,
    end_value: Optional[float] = None,
    non_pad_mask: Optional[Tensor] = None,
    lengths: Optional[Tensor] = None,
) -> list[Tensor]:
    """
    You must provide a value for one of pad_value, end_value non_pad_mask or lengths.
    If multiple values are provided, only one will be used and the priority order is [pad_value, end_value non_pad_mask, lengths].

    :param tensor: (N, *)
    :param pad_value: TODO
    :param end_value: TODO
    :param non_pad_mask: TODO
    :param lengths: TODO
    :returns: A list of N tensors of shape (*)
    """

    if pad_value is not None:
        lengths = tensor_to_lengths(tensor, pad_value=pad_value)
        return tensor_to_tensors_list(tensor, lengths=lengths)

    elif end_value is not None:
        lengths = tensor_to_lengths(tensor, end_value=end_value)
        return tensor_to_tensors_list(tensor, lengths=lengths)

    elif non_pad_mask is not None:
        lengths = non_pad_mask_to_lengths(non_pad_mask)
        return tensor_to_tensors_list(tensor, lengths=lengths)

    elif lengths is not None:
        slices_lst = [
            [slice(None) for _ in range(tensor.ndim)] + [slice(0, len_)]
            for len_ in lengths
        ]
        tensors = [tensor[slices] for slices in slices_lst]

    else:
        raise ValueError(
            "Invalid arguments. Please provide only one of the arguments : end_value, pad_value, mask, lengths."
        )

    return tensors


def tensor_to_lengths(
    tensor: Tensor,
    pad_value: Optional[float] = None,
    end_value: Optional[float] = None,
    dim: int = -1,
) -> Tensor:
    """
    You must provide a value for one of pad_value or end_value. If both values are provided, the end_value is ignored.

    :param tensor: Tensor of shape (N, *)
    :param pad_value: The pad value used in tensor. defaults to None.
    :param end_value: The end value used in tensor. defaults to None.
    :param dim: TODO
    :returns: The lengths as IntTensor of shape (N,)
    """
    if pad_value is not None:
        non_pad_mask = tensor != pad_value
        lengths = non_pad_mask.sum(dim=dim)

    elif end_value is not None:
        contains_eos = (tensor == end_value).any(dim=dim)
        indexes_eos = (tensor == end_value).int().argmax(dim=dim)
        lengths = torch.where(contains_eos, indexes_eos, tensor.shape[dim])

    else:
        raise ValueError(
            "Invalid arguments. Please provide only one of the arguments : end_value, pad_value."
        )

    return lengths


def tensor_to_non_pad_mask(
    tensor: Tensor,
    pad_value: Optional[float] = None,
    end_value: Optional[float] = None,
) -> Tensor:
    """Convert tensor to non-pad binary mask.
    You must provide a value for one of pad_value or end_value. If both values are provided, the end_value is ignored.

    :param tensor: A tensor of values. If end_value is given instead of pad_value, the number of dims must be <= 2.
    :param pad_value: The pad value used in tensor. defaults to None.
    :param end_value: The end value used in tensor. defaults to None.
    :returns: A binary mask representing the non-padded values. Shape is the same than the input tensor.
    """
    if pad_value is not None:
        non_pad_mask = tensor.ne(pad_value)

    elif end_value is not None:
        if tensor.ndim > 2:
            raise ValueError(
                f"Cannot compute non_pad_mask for with more than 2 dimensions with {end_value=}. (found {tensor.ndim=})"
            )
        lengths = tensor_to_lengths(tensor, end_value=end_value, dim=-1)
        non_pad_mask = lengths_to_non_pad_mask(lengths, tensor.shape[-1], False)

    else:
        raise ValueError(
            "Invalid arguments. Please provide only one of the arguments : end_value, pad_value."
        )

    return non_pad_mask


def tensor_to_pad_mask(
    tensor: Tensor,
    pad_value: Optional[float] = None,
    end_value: Optional[float] = None,
) -> Tensor:
    """Convert tensor to pad binary mask.
    You must provide a value for one of pad_value or end_value. If both values are provided, the end_value is ignored.

    :param tensor: A tensor of values. If end_value is given instead of pad_value, the number of dims must be <= 2.
    :param pad_value: The pad value used in tensor. defaults to None.
    :param end_value: The end value used in tensor. defaults to None.
    :returns: A binary mask representing the padded values. Shape is the same than the input tensor.
    """
    non_pad_mask = tensor_to_non_pad_mask(tensor, pad_value, end_value)
    return non_pad_mask.logical_not()


def non_pad_mask_to_lengths(mask: Tensor, dim: int = -1) -> Tensor:
    return mask.sum(dim=dim)


def pad_mask_to_lengths(mask: Tensor, dim: int = -1) -> Tensor:
    return mask.shape[dim] - non_pad_mask_to_lengths(mask, dim)


def lengths_to_non_pad_mask(
    lengths: Tensor,
    max_len: Optional[int],
    include: bool = True,
) -> Tensor:
    """Convert lengths to binary mask of non-padded values.

    :param lengths: (bsize,)
    :param max_len: Optional int for indicate the maximal length.
        If None, it will be set to lengths.max().
        defaults to None.
    :param include: If True, the value at index of len will be True in returned value.
        defaults to True.
    :returns: (bsize, max_len)
    """
    dim = -1
    if max_len is None:
        max_len = int(lengths.max(dim=dim)[0].item())
    indices = torch.arange(0, max_len, device=lengths.device) + 1
    lengths = lengths.unsqueeze(dim=-1)
    non_pad_mask = indices < lengths if not include else indices <= lengths
    return non_pad_mask


def lengths_to_pad_mask(
    lengths: Tensor,
    max_len: Optional[int],
    include: bool = False,
) -> Tensor:
    """Convert lengths to binary mask of padded values.

    :param lengths: (bsize,)
    :param max_len: Optional int for indicate the maximal length.
        If None, it will be set to lengths.max().
        defaults to None.
    :param include: If True, the value at index of len will be True in returned value.
        defaults to False.
    :returns: (bsize, max_len)
    """
    non_pad_mask = lengths_to_non_pad_mask(lengths, max_len, not include)
    return non_pad_mask.logical_not()


def pad_last_dim(tensor: Tensor, target_length: int, pad_value: float) -> Tensor:
    pad_len = max(target_length - tensor.shape[-1], 0)
    return F.pad(tensor, [0, pad_len], value=pad_value)


def pad_sequence_rec(
    sequence: Union[Tensor, int, float, tuple, list],
    pad_value: float,
    dtype: Union[None, torch.dtype] = None,
    device: Union[None, torch.device] = None,
) -> Tensor:
    """Recursive version of torch.nn.utils.rnn.pad_sequence, with padding of Tensors.

    :param sequence: The sequence to pad. Must be convertable to tensor by having the correct number of dims in all sublists.
    :param pad_value: The pad value used.
    :param dtype: The dtype of the output Tensor. defaults to None.
    :param device: The device of the output Tensor. defaults to None.
    :returns: The sequence as a padded Tensor.

    Example 1
    ----------
    >>> sequence = [[1, 2], [3], [], [4, 5]]
    >>> output = pad_sequence_rec(sequence, 0)
    tensor([[1, 2], [3, 0], [0, 0], [4, 5]])

    Example 2
    ----------
    >>> invalid_sequence = [[1, 2, 3], 3]
    >>> output = pad_sequence_rec(invalid_sequence, 0)
    ValueError : Cannot pad sequence of tensors of differents number of dims.

    """
    if isinstance(sequence, Tensor):
        return sequence.to(dtype=dtype, device=device)

    if isinstance(sequence, (int, float)) or (
        isinstance(sequence, Sized) and len(sequence) == 0
    ):
        return torch.as_tensor(sequence, dtype=dtype, device=device)  # type: ignore

    elif isinstance(sequence, (list, tuple)):
        if all(isinstance(elt, (int, float)) for elt in sequence):
            return torch.as_tensor(sequence, dtype=dtype, device=device)  # type: ignore

        sequence = [pad_sequence_rec(elt, pad_value, dtype, device) for elt in sequence]
        # sequence is now a list[Tensor]
        shapes = [elt.shape for elt in sequence]

        # If all tensors have the same shape
        if all(shape == shapes[0] for shape in shapes):
            return torch.stack(sequence, dim=0)

        # If all tensors have the same number of dims
        elif all(elt.ndim == sequence[0].ndim for elt in sequence):
            if all(shape[1:] == shapes[0][1:] for shape in shapes):
                return pad_sequence(sequence, True, pad_value)
            else:
                max_lens = [
                    max(shape[i] for shape in shapes) for i in range(sequence[0].ndim)
                ]
                paddings = [
                    [
                        (max_lens[i] - elt.shape[i]) * j
                        for i in range(-1, -sequence[0].ndim, -1)
                        for j in range(2)
                    ]
                    for elt in sequence
                ]
                sequence = [
                    F.pad(elt, padding, value=pad_value)
                    for elt, padding in zip(sequence, paddings)
                ]
                return pad_sequence(sequence, True, pad_value)

        else:
            raise ValueError(
                f"Cannot pad sequence of tensors of differents number of dims. ({sequence=}, {shapes=})"
            )

    else:
        raise TypeError(
            f"Invalid type {type(sequence)}. (expected Tensor, int, float, list or tuple)"
        )


def stack_tensors_rec(
    sequence: Union[Tensor, int, float, tuple, list],
    relaxed: bool = False,
    dtype: Union[None, torch.dtype] = None,
    device: Union[None, str, torch.device] = None,
) -> Union[Tensor, list]:
    if isinstance(device, str):
        device = torch.device(device)

    def stack_tensors_rec_relaxed(
        sequence: Union[Tensor, int, float, tuple, list]
    ) -> Union[Tensor, list]:
        if isinstance(sequence, Tensor):
            return sequence.to(dtype=dtype, device=device)
        elif isinstance(sequence, (int, float)):
            return torch.as_tensor(sequence, dtype=dtype, device=device)  # type: ignore
        elif isinstance(sequence, (list, tuple)):
            if all(isinstance(elt, (int, float)) for elt in sequence):
                return torch.as_tensor(sequence, dtype=dtype, device=device)  # type: ignore

            sequence = [stack_tensors_rec_relaxed(elt) for elt in sequence]
            if all(isinstance(elt, Tensor) for elt in sequence):
                shapes = [elt.shape for elt in sequence]  # type: ignore
                if len(sequence) == 0 or all(shape == shapes[0] for shape in shapes):
                    return torch.stack(sequence)  # type: ignore
                elif relaxed:
                    return sequence
                else:
                    raise ValueError(
                        f"Cannot stack tensors of different shapes. (found {shapes=})"
                    )
            elif relaxed:
                return sequence
            else:
                raise ValueError("Cannot stack tensors of different shape or types.")
        else:
            raise TypeError(
                f"Invalid type {type(sequence)}. (expected Tensor, int, float, list or tuple)"
            )

    sequence = stack_tensors_rec_relaxed(sequence)
    return sequence


def softmax_multidim(
    x: Tensor,
    dims: tuple[int, ...],
    clamp_min: float = 0.0,
) -> Tensor:
    x_exp = x.exp()
    return x_exp / x_exp.sum(dim=dims, keepdim=True).clamp_min_(min=clamp_min)


def batch_conv2d_naive(x: Tensor, weight: Tensor) -> Tensor:
    """
    Conv2d with a batch of distincts weights. (slow version using Conv2d multiple times)

    :param x: (bsize, in_channels, x_width, x_height)
    :param weight: (bsize, out_channels, in_channels, weight_width, weight_height)
    :returns: (bsize, out_channels, x_width, x_height)
    """
    if (
        x.ndim != 4
        or weight.ndim != 5
        or x.shape[0] != weight.shape[0]
        or x.shape[1] != weight.shape[2]
    ):
        raise ValueError(
            f"Invalid arguments for batch_conv2d_naive. ({x.shape=}; {weight.shape=})"
        )

    x = torch.stack(
        [
            F.conv2d(x_i.unsqueeze(dim=0), weight=w_i, bias=None, padding="same")
            for x_i, w_i in zip(x, weight)
        ]
    )
    x = x.squeeze(dim=1)
    return x.contiguous()


def batch_conv2d(x: Tensor, weight: Tensor) -> Tensor:
    """
    Conv2d with a batch of distincts weights. (faster version using only 1 Conv2d with groups)

    :param x: (bsize, in_channels, x_width, x_height)
    :param weight: (bsize, out_channels, in_channels, weight_width, weight_height)
    :returns: (bsize, out_channels, x_width, x_height)
    """
    if (
        x.ndim != 4
        or weight.ndim != 5
        or x.shape[0] != weight.shape[0]
        or x.shape[1] != weight.shape[2]
    ):
        raise ValueError(
            f"Invalid arguments for batch_conv2d. ({x.shape=}; {weight.shape=})"
        )

    x_width, x_height = x.shape[2:]
    bsize, out_channels, in_channels, weight_width, weight_height = weight.shape
    x = x.view(1, bsize * in_channels, x_width, x_height).contiguous()
    weight = weight.view(
        bsize * out_channels, in_channels, weight_width, weight_height
    ).contiguous()
    x = F.conv2d(x, weight=weight, bias=None, padding="same", groups=bsize)
    x = x.view(bsize, out_channels, x_width, x_height)
    return x.contiguous()


def pad_sequence_1d(tensors: list[Tensor], pad_value: float) -> Tensor:
    if not all(tensor.ndim == 1 for tensor in tensors):
        raise ValueError("Invalid argument tensors for pad_sequence_1d.")

    max_len = max(tensor.shape[0] for tensor in tensors)
    output = torch.empty(
        (len(tensors), max_len), device=tensors[0].device, dtype=tensors[0].dtype
    )
    for i, tensor in enumerate(tensors):
        output[i, : tensor.shape[0]] = tensor
        output[i, tensor.shape[0] :] = pad_value
    return output


def pad_sequence_nd(tensors: list[Tensor], pad_value: float) -> Tensor:
    if not all(tensor.ndim >= 1 for tensor in tensors):
        raise ValueError("Invalid argument tensors for pad_sequence_1d.")
    if not all(tensor.shape[1:] == tensors[0].shape[1:] for tensor in tensors[1:]):
        raise ValueError("Invalid argument tensors for pad_sequence_1d.")

    max_len = max(tensor.shape[0] for tensor in tensors)
    output = torch.empty(
        (len(tensors), max_len) + tuple(tensors[0].shape[1:]),
        device=tensors[0].device,
        dtype=tensors[0].dtype,
    )
    for i, tensor in enumerate(tensors):
        output[i, : tensor.shape[0]] = tensor
        output[i, tensor.shape[0] :] = pad_value
    return output


def move_to_rec(
    x: Any,
    *args,
    predicate: Optional[Callable[[Any], bool]] = None,
    **kwargs,
) -> Any:
    if isinstance(x, (Tensor, nn.Module)):
        if predicate is None or predicate(x):
            return x.to(*args, **kwargs)
        else:
            return x
    elif isinstance(x, (str,)):
        return x
    elif isinstance(x, Iterable):
        if isinstance(x, dict):
            return dict(move_to_rec(x.items(), predicate=predicate, *args, **kwargs))
        else:
            generator = (
                move_to_rec(xi, predicate=predicate, *args, **kwargs) for xi in x
            )
            if isinstance(x, Generator):
                return generator
            elif isinstance(x, tuple):
                return tuple(generator)
            elif isinstance(x, list):
                return list(generator)
            else:
                return list(generator)
    else:
        return x


def get_inverse_perm(indexes: Tensor, dim: int = -1) -> Tensor:
    arange = torch.arange(
        indexes.shape[dim],
        dtype=indexes.dtype,
        device=indexes.device,
    )
    arange = arange.expand(*indexes.shape)

    indexes_inv = torch.empty_like(indexes)
    indexes_inv = indexes_inv.scatter(dim, indexes, arange)
    return indexes_inv


def sample_gumbels_like(
    tensor: Tensor,
    generator: Union[None, torch.Generator] = None,
) -> Tensor:
    gumbels = (
        -torch.empty_like(tensor, memory_format=torch.legacy_contiguous_format)
        .exponential_(generator=generator)
        .log()
    )  # ~Gumbel(0,1)
    return gumbels


def gumbel_log_softmax(
    logits: Tensor,
    tau: float = 1.0,
    hard: bool = False,
    dim: int = -1,
    generator: Union[None, torch.Generator] = None,
) -> Tensor:
    """
    BASED on https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax
    But has generator arg
    """
    gumbels = sample_gumbels_like(logits, generator)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.log_softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


def gumbel_softmax(
    logits: Tensor,
    tau: float = 1.0,
    hard: bool = False,
    dim: int = -1,
    generator: Union[None, torch.Generator] = None,
) -> Tensor:
    """
    BASED on https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax
    But has generator arg and compute softmax instead of log_softmax.
    """
    gumbels = sample_gumbels_like(logits, generator)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret
