#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import sys

from typing import Any, Iterable, Optional, Sized, Union

import torch

from torch import nn, Tensor

from dcase2022task6a.tokenization.constants import EOS_TOKEN, SOS_TOKEN, UNK_TOKEN
from dcase2022task6a.tokenization.normalizers import (
    CleanDoubleSpaces,
    CleanPunctuation,
    CleanSpacesBeforePunctuation,
    CleanSpecialTokens,
    Lowercase,
    NormalizerI,
    Strip,
)
from dcase2022task6a.tokenization.tokenizers import get_pre_tokenizer_with_level


class AACTokenizer(nn.Module):
    """Provide function to encode/decode sentences to Tensor.

    Contains normalizers that clean sentences, TokenizerI to split to tokens and token to index dictionaries.
    """

    OUT_TYPES = ("str", "int", "Tensor", "pt")

    def __init__(
        self,
        level: str = "word",
        lowercase: bool = True,
        clean_punctuation: bool = True,
        **kwargs,
    ) -> None:
        """
        :param level: "word", "char", "affix", "bpe", "unigram" or "bert".
        :param lowercase: If True, encoded sentences will be converted to lowercase. defaults to True.
        :param clean_punctuation: If True, clean punctuation tokens. defaults to True.
        :param **kwargs: These optional values passed to the internal pre_tokenizer. The accepted values depends of the "level" argument.
            level == "affix":
                language: str = "english" (stemmer language)
                kwargs: Any arguments of level == "word" because it starts to tokenize to words before splitting into affixes.
            level == "bert":
                model_name: str = "bert-base-uncased"
            level == "bpe" or level == "unigram":
                vocab_size: int = 1000
                split_by_whitespace: bool = True
                character_coverage: float = 0.9995
                verbose: int = 1
            level == "char": (no accepted values)
            level == "word":
                backend: str = "spacy" (one of "spacy", "nltk", "ptb", "python")
                backend == "spacy":
                    model_name: str = "en_core_web_sm"
                backend == "nltk":
                    model_name: str = "english"
                backend == "ptb"
                    java_path: str = "java"
                    ext_dpath: str = "ext"
                    tmp_dpath: str = "/tmp"
                backend == "python"
                    separator: str
        """
        super().__init__()

        hparams = {
            "level": level,
            "lowercase": lowercase,
            "clean_punctuation": clean_punctuation,
        } | kwargs

        # Build normalizers
        pre_encoding_normalizers: list[NormalizerI] = [
            CleanSpecialTokens(),
        ]
        if lowercase:
            pre_encoding_normalizers.append(Lowercase())
        if clean_punctuation:
            pre_encoding_normalizers.append(CleanPunctuation())
        else:
            pre_encoding_normalizers.append(CleanSpacesBeforePunctuation())
        pre_encoding_normalizers += [
            CleanDoubleSpaces(),
            Strip(),
        ]

        post_decoding_normalizers: list[NormalizerI] = [
            CleanSpecialTokens(),
            CleanSpacesBeforePunctuation(),
            CleanDoubleSpaces(),
            Strip(),
        ]
        if lowercase:
            post_decoding_normalizers.append(Lowercase())

        # Set attributes
        self._hparams = hparams
        self._max_sentence_size = -1
        self._min_sentence_size = sys.maxsize
        self._n_sentences_fit = 0
        self._itos = {}
        self._stoi = {}
        self._vocab = {}
        self._pre_encoding_normalizers = pre_encoding_normalizers
        self._post_decoding_normalizers = post_decoding_normalizers
        self._tokenizer = get_pre_tokenizer_with_level(
            level=level,
            **kwargs,
        )

    def decode_batch(self, sentences: Union[Tensor, Iterable]) -> list[str]:
        if not isinstance(sentences, Sized):
            sentences = list(sentences)

        if len(sentences) == 0:
            return []

        elif all(
            (isinstance(token, Tensor) and token.ndim == 0)
            for sentence in sentences
            for token in sentence
        ):
            sentences = [[token.item() for token in sentence] for sentence in sentences]
            return self.decode_batch(sentences)

        elif all(
            isinstance(token, int) for sentence in sentences for token in sentence
        ):
            sentences = [
                [self.itos(token) for token in sentence] for sentence in sentences
            ]
            return self.decode_batch(sentences)

        elif all(
            isinstance(token, str) for sentence in sentences for token in sentence
        ):
            sentences = self._tokenizer.detokenize_batch(sentences)  # type: ignore
            for normalizer in self._post_decoding_normalizers:
                sentences = normalizer.normalize_batch(sentences)
            return sentences

        else:
            raise TypeError(
                "Invalid sentences type in decode_batch method. (expected Tensor with ndim=2, list[list[str]] or list[list[int]])"
            )

    def decode_rec(self, nested_sentences: Union[Tensor, Iterable]) -> Union[str, list]:
        if not (
            (isinstance(nested_sentences, Tensor) and nested_sentences.ndim > 0)
            or isinstance(nested_sentences, Iterable)
        ):
            raise TypeError(
                f"Invalid {nested_sentences=} for decode_rec method in {self.__class__.__name__}. (expected a Tensor of ndim > 0 or a list)"
            )

        if isinstance(nested_sentences, Tensor):
            return self.decode_rec(nested_sentences.tolist())
        elif _is_encoded_sentence(nested_sentences):
            return self.decode_single(nested_sentences)
        elif isinstance(nested_sentences, Iterable) and all(
            map(_is_encoded_sentence, nested_sentences)
        ):
            return self.decode_batch(nested_sentences)
        else:
            return [self.decode_rec(sentences) for sentences in nested_sentences]

    def decode_single(self, sentence: Union[Tensor, Iterable]) -> str:
        return self.decode_batch([sentence])[0]

    def encode_batch(
        self,
        sentences: Iterable[str],
        add_sos_eos: bool = False,
        out_type: str = "str",
        unk_token: Optional[str] = UNK_TOKEN,
    ) -> Union[list, Tensor]:
        for normalizer in self._pre_encoding_normalizers:
            sentences = normalizer.normalize_batch(sentences)

        tokenized_sentences = self._tokenizer.tokenize_batch(sentences)
        del sentences

        if add_sos_eos:
            tokenized_sentences = [
                [SOS_TOKEN] + sentence + [EOS_TOKEN] for sentence in tokenized_sentences
            ]

        if out_type == "str":
            pass
        elif out_type in ("int", "Tensor", "pt"):
            if unk_token is None:
                invalid_tokens = [
                    token
                    for sentence in tokenized_sentences
                    for token in sentence
                    if token not in self._stoi
                ]
                if len(invalid_tokens) > 0:
                    raise ValueError(
                        f"Invalid sentences tokens (found tokens {invalid_tokens} not in vocabulary from {tokenized_sentences=}, {add_sos_eos=}, {out_type=}, {unk_token=})."
                    )

            tokenized_sentences = [
                [self.stoi(token, unk_token) for token in sentence]
                for sentence in tokenized_sentences
            ]
            if out_type in ("Tensor", "pt"):
                tokenized_sentences = [
                    torch.as_tensor(sentence) for sentence in tokenized_sentences
                ]
                if len(tokenized_sentences) == 0 or all(
                    sentence.shape == tokenized_sentences[0].shape
                    for sentence in tokenized_sentences
                ):
                    tokenized_sentences = torch.stack(tokenized_sentences)
        else:
            raise ValueError(
                f"Invalid argument {out_type=}. (expected one of {AACTokenizer.OUT_TYPES})"
            )

        return tokenized_sentences

    def encode_rec(
        self,
        nested_sentences: Union[str, Iterable],
        add_sos_eos: bool = False,
        out_type: str = "str",
        unk_token: Optional[str] = UNK_TOKEN,
    ) -> Union[list, Tensor]:
        if isinstance(nested_sentences, str):
            return self.encode_single(
                nested_sentences, add_sos_eos, out_type, unk_token
            )
        elif all(isinstance(sentence, str) for sentence in nested_sentences):
            return self.encode_batch(nested_sentences, add_sos_eos, out_type, unk_token)
        else:
            return [
                self.encode_rec(sentences, add_sos_eos, out_type, unk_token)
                for sentences in nested_sentences
            ]

    def encode_single(
        self,
        sentence: str,
        add_sos_eos: bool = False,
        out_type: str = "str",
        unk_token: Optional[str] = UNK_TOKEN,
    ) -> Union[list, Tensor]:
        return self.encode_batch([sentence], add_sos_eos, out_type, unk_token)[0]

    def extra_repr(self) -> str:
        return ", ".join(
            f"{name}={value}" for name, value in self.get_hparams().items()
        )

    def fit(self, sentences: Iterable[str]) -> tuple[list, dict, dict, dict]:
        if self._n_sentences_fit > 0:
            raise RuntimeError(
                f"Cannot fit {self.__class__.__name__} twice. (found n_sentences_fit={self._n_sentences_fit} > 0)"
            )

        for normalizer in self._pre_encoding_normalizers:
            sentences = normalizer.normalize_batch(sentences)

        encoded_sentences, itos, stoi, vocab = self._tokenizer.fit(sentences)
        del sentences

        self._itos |= itos
        self._stoi |= stoi
        self._vocab |= vocab

        sentences_lens = list(map(len, encoded_sentences))
        self._max_sentence_size = max(self._max_sentence_size, max(sentences_lens))
        self._min_sentence_size = min(self._min_sentence_size, min(sentences_lens))
        self._n_sentences_fit += len(encoded_sentences)

        return encoded_sentences, itos, stoi, vocab

    def forward(self, inputs: Any, *args, **kwargs) -> Any:
        return self.encode_rec(inputs, *args, **kwargs)

    def get_hparams(self) -> dict[str, Any]:
        return self._hparams

    def get_level(self) -> str:
        return self._tokenizer.get_level()

    def get_max_sentence_size(self) -> int:
        return self._max_sentence_size

    def get_min_sentence_size(self) -> int:
        return self._min_sentence_size

    def get_vocab(self) -> dict[str, int]:
        """Returns the vocabulary with the number of occurrence of each token in sentences fit."""
        return self._vocab

    def get_vocab_size(self) -> int:
        """Returns the size of the vocabulary, i.e. `len(tokenizer.get_vocab())`."""
        return len(self.get_vocab())

    def has(self, token: str) -> bool:
        return token in self._vocab.keys()

    def is_fit(self) -> bool:
        return self._n_sentences_fit > 0

    def itos(self, index: Union[int, Tensor]) -> str:
        if isinstance(index, Tensor):
            if index.ndim != 0 or index.is_floating_point():
                raise ValueError(
                    f"Invalid argument {index=}. (expected an int or a scalar integer tensor)"
                )
            index = int(index.item())
        return self._itos[index]

    @classmethod
    def load(cls, fpath: str) -> "AACTokenizer":
        with open(fpath, "rb") as file:
            tokenizer = pickle.load(file)
        return tokenizer

    def save(self, fpath: str) -> None:
        with open(fpath, "wb") as file:
            pickle.dump(self, file)

    def stoi(self, token: str, default: Optional[str] = UNK_TOKEN) -> int:
        """Returns the correponding id of a token.

        If default is None and token is not in the tokenizer vocabulary, raises a KeyError.
        If default is not None and token is not in the tokenizer vocabulary, returns stoi(default).
        Otherwise returns the id of the token.

        :param token: The input token.
        :param default: If not None, it is the default value of the token.
        :returns: The id of the token.
        """
        if default is None:
            return self._stoi[token]
        elif default in self._stoi:
            return self._stoi.get(token, self._stoi[default])
        else:
            raise KeyError(f"Invalid default value {default=}.")

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, AACTokenizer):
            return False
        return pickle.dumps(self) == pickle.dumps(__o)

    def __getitem__(self, input_: Union[int, str]) -> Union[int, str]:
        if isinstance(input_, int):
            return self.itos(input_)
        elif isinstance(input_, str):
            return self.stoi(input_)
        else:
            raise TypeError(
                f"Invalid input type {input_.__class__.__name__}. (expected int or str)"
            )

    def __getstate__(self) -> dict[str, Any]:
        # avoid store backward hooks, nn.Module attributes...
        module_dict = nn.Module().__dict__
        tokenizer_dict = {
            k: v for k, v in self.__dict__.items() if k not in module_dict
        }
        state = {
            "_target_": _get_full_class_name(self),
            "tokenizer": tokenizer_dict,
        }
        return state

    def __hash__(self) -> int:
        return sum(pickle.dumps(self))

    def __len__(self) -> int:
        return self.get_vocab_size()

    def __setstate__(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict) or "tokenizer" not in state.keys():
            raise TypeError(
                f"Incompatible pickle value type {type(state)}. (expected dict with key 'tokenizer')"
            )
        tokenizer_dict = state["tokenizer"]
        self.__dict__.update(nn.Module().__dict__)
        self.__dict__.update(tokenizer_dict)


def _get_full_class_name(obj: Any) -> str:
    """Returns the classname of an object with parent modules.

    Example 1
    ----------
    >>> _get_obj_fullname(torch.nn.Linear(10, 10))
    'torch.nn.modules.linear.Linear'
    """
    class_ = obj.__class__
    module = class_.__module__
    if module == "builtins":
        # avoid outputs like 'builtins.str'
        return class_.__qualname__
    return module + "." + class_.__qualname__


def _is_encoded_sentence(inputs: Any) -> bool:
    """Returns true if inputs is:
    - list[int]
    - list[str]
    - list[Tensor with ndim=0]
    - Tensor with ndim=1
    """
    return (
        isinstance(inputs, list)
        and all(
            isinstance(input_, (str, int))
            or (isinstance(input_, Tensor) and input_.ndim == 0)
            for input_ in inputs
        )
    ) or (isinstance(inputs, Tensor) and inputs.ndim == 1)
