#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import torch

from torch import nn, Tensor


class PositionalEncoding(nn.Module):
    # BASED ON PYTORCH TUTORIAL : https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(
        self,
        emb_size: int,
        dropout: float,
        maxlen: int = 5000,
    ) -> None:
        super().__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor) -> Tensor:
        pos_embedding_value = self.pos_embedding[: token_embedding.size(0), :]
        return self.dropout(token_embedding + pos_embedding_value)
