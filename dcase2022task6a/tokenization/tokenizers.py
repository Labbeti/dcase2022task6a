#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import logging
import os
import os.path as osp
import subprocess
import tempfile
import time

from abc import ABC
from collections import Counter
from typing import Any, Hashable, Iterable, Optional

import sentencepiece as sp
import spacy

from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from transformers.models.bert.tokenization_bert import BertTokenizer

from dcase2022task6a.tokenization.constants import (
    SOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
    SPECIAL_TOKENS,
)


# path to the stanford corenlp jar
STANFORD_CORENLP_3_4_1_JAR = osp.join("stanford_nlp", "stanford-corenlp-3.4.1.jar")

# punctuations to be removed from the sentences
PTB_PUNCTUATIONS = (
    "''",
    "'",
    "``",
    "`",
    "-LRB-",
    "-RRB-",
    "-LCB-",
    "-RCB-",
    ".",
    "?",
    "!",
    ",",
    ":",
    "-",
    "--",
    "...",
    ";",
)


class TokenizerI(ABC):
    def detokenize_batch(self, sentences: Iterable[Iterable[str]]) -> list[str]:
        raise NotImplementedError("Abstract method")

    def fit(self, sentences: Iterable[str]) -> tuple[list, dict, dict, dict]:
        raise NotImplementedError("Abstract method")

    def get_level(self) -> str:
        raise NotImplementedError("Abstract method")

    def tokenize_batch(self, sentences: Iterable[str]) -> list[list[str]]:
        raise NotImplementedError("Abstract method")


class WordTokenizer(TokenizerI):
    BACKENDS = ("spacy", "nltk", "ptb", "python")

    def __init__(self, backend: str = "spacy", *args, **kwargs) -> None:
        if backend == "spacy":
            tokenizer = SpacyWordTokenizer(*args, **kwargs)
        elif backend == "nltk":
            tokenizer = NLTKWordTokenizer(*args, **kwargs)
        elif backend == "ptb":
            tokenizer = PTBWordTokenizer(*args, **kwargs)
        elif backend == "python":
            tokenizer = PythonWordTokenizer(*args, **kwargs)
        else:
            raise ValueError(
                f"Invalid argument {backend=}. (expected one of {WordTokenizer.BACKENDS})"
            )

        super().__init__()
        self._tokenizer = tokenizer

    def detokenize_batch(self, sentences: Iterable[Iterable[str]]) -> list[str]:
        return self._tokenizer.detokenize_batch(sentences)

    def fit(self, sentences: Iterable[str]) -> tuple[list, dict, dict, dict]:
        return self._tokenizer.fit(sentences)

    def get_level(self) -> str:
        return self._tokenizer.get_level()

    def tokenize_batch(self, sentences: Iterable[str]) -> list[list[str]]:
        return self._tokenizer.tokenize_batch(sentences)


class SpacyWordTokenizer(TokenizerI):
    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        super().__init__()
        self._model_name = model_name
        self._model = spacy.load(model_name)

    def detokenize_batch(self, sentences: Iterable[Iterable[str]]) -> list[str]:
        decoded_sentences = [" ".join(sentence) for sentence in sentences]
        return decoded_sentences

    def fit(self, sentences: Iterable[str]) -> tuple[list, dict, dict, dict]:
        encoded_sentences = self.tokenize_batch(sentences)
        itos, stoi, vocab = _build_mappings_and_vocab(encoded_sentences)
        return encoded_sentences, itos, stoi, vocab

    def get_level(self) -> str:
        return "word"

    def tokenize_batch(self, sentences: Iterable[str]) -> list[list[str]]:
        encoded_sentences = [self._model.tokenizer(sentence) for sentence in sentences]
        # Note : Spacy returns a list of spacy.tokens.token.Token object
        encoded_sentences = [
            [word.text for word in sentence] for sentence in encoded_sentences
        ]
        return encoded_sentences

    def __getstate__(self) -> dict[str, Any]:
        return {"model_name": self._model_name}

    def __setstate__(self, data: dict[str, Any]) -> None:
        model_name = data["model_name"]
        self._model_name = model_name
        self._model = spacy.load(model_name)


class PTBWordTokenizer(TokenizerI):
    def __init__(
        self,
        java_path: str = "java",
        ext_dpath: str = "ext",
        tmp_dpath: str = "/tmp",
    ) -> None:
        super().__init__()
        self._java_path = java_path
        self._ext_dpath = ext_dpath
        self._tmp_dpath = tmp_dpath

    def detokenize_batch(self, sentences: Iterable[Iterable[str]]) -> list[str]:
        decoded_sentences = [" ".join(sentence) for sentence in sentences]
        return decoded_sentences

    def fit(self, sentences: Iterable[str]) -> tuple[list, dict, dict, dict]:
        encoded_sentences = self.tokenize_batch(sentences)
        itos, stoi, vocab = _build_mappings_and_vocab(encoded_sentences)
        return encoded_sentences, itos, stoi, vocab

    def get_level(self) -> str:
        return "word"

    def tokenize_batch(self, sentences: Iterable[str], **kwargs) -> list[list[str]]:
        return _ptb_tokenize(
            sentences,
            java_path=self._java_path,
            ext_dpath=self._ext_dpath,
            tmp_dpath=self._tmp_dpath,
        )


class PythonWordTokenizer(TokenizerI):
    def __init__(self, separator: str = " ") -> None:
        super().__init__()
        self._separator = separator

    def detokenize_batch(self, sentences: Iterable[Iterable[str]]) -> list[str]:
        decoded_sentences = [" ".join(sentence) for sentence in sentences]
        return decoded_sentences

    def fit(self, sentences: Iterable[str]) -> tuple[list, dict, dict, dict]:
        encoded_sentences = self.tokenize_batch(sentences)
        itos, stoi, vocab = _build_mappings_and_vocab(encoded_sentences)
        return encoded_sentences, itos, stoi, vocab

    def get_level(self) -> str:
        return "word"

    def tokenize_batch(self, sentences: Iterable[str], **kwargs) -> list[list[str]]:
        return [sentence.split(self._separator) for sentence in sentences]


def get_pre_tokenizer_with_level(level: str, *args, **kwargs) -> TokenizerI:
    if level == "word":
        return WordTokenizer(*args, **kwargs)
    else:
        LEVELS = ("word",)
        raise ValueError(f"Invalid argument {level=}. (expected one of {LEVELS})")


def _build_mappings_and_vocab(
    encoded_sentences: list[list[str]],
    add_special_tokens: bool = True,
) -> tuple[dict, dict, dict]:
    """Returns (itos, stoi, vocab) dictionaries."""
    tokens_counter = {}
    if add_special_tokens:
        tokens_counter |= {token: 0 for token in SPECIAL_TOKENS}
    tokens_counter |= dict(
        Counter(token for sentence in encoded_sentences for token in sentence)
    )
    itos = {i: token for i, token in enumerate(tokens_counter.keys())}
    stoi = {token: i for i, token in enumerate(tokens_counter.keys())}
    return itos, stoi, tokens_counter


def _ptb_tokenize(
    sentences: Iterable[str],
    audio_ids: Optional[Iterable[Hashable]] = None,
    java_path: str = "java",
    ext_dpath: str = "ext",
    tmp_dpath: str = "/tmp",
    verbose: int = 1,
) -> list[list[str]]:
    # Based on https://github.com/audio-captioning/caption-evaluation-tools/blob/c1798df4c91e29fe689b1ccd4ce45439ec966417/coco_caption/pycocoevalcap/tokenizer/ptbtokenizer.py#L30
    sentences = list(sentences)
    if len(sentences) == 0:
        return []

    stanford_fpath = osp.join(ext_dpath, STANFORD_CORENLP_3_4_1_JAR)

    if not osp.isdir(ext_dpath):
        raise RuntimeError(f"Cannot find ext directory at {ext_dpath=}.")
    if not osp.isdir(tmp_dpath):
        raise RuntimeError(f"Cannot find tmp directory at {tmp_dpath=}.")
    if not osp.isfile(stanford_fpath):
        raise FileNotFoundError(
            f"Cannot find jar file {STANFORD_CORENLP_3_4_1_JAR} in {ext_dpath=}."
        )
    start_time = time.perf_counter()
    if verbose >= 2:
        logging.debug(
            f"Start executing {STANFORD_CORENLP_3_4_1_JAR} JAR file for tokenization. ({len(sentences)=})"
        )

    cmd = [
        java_path,
        "-cp",
        stanford_fpath,
        "edu.stanford.nlp.process.PTBTokenizer",
        "-preserveLines",
        "-lowerCase",
    ]

    # ======================================================
    # prepare data for PTB AACTokenizer
    # ======================================================
    if audio_ids is None:
        audio_ids = list(range(len(sentences)))
    else:
        audio_ids = list(audio_ids)

    if len(audio_ids) != len(sentences):
        raise ValueError(
            f"Invalid number of audio ids ({len(audio_ids)}) with sentences len={len(sentences)}."
        )

    sentences = "\n".join(sentences)

    # ======================================================
    # save sentences to temporary file
    # ======================================================
    path_to_jar_dirname = tmp_dpath
    tmp_file = tempfile.NamedTemporaryFile(
        delete=False,
        dir=path_to_jar_dirname,
        suffix=".txt",
    )
    tmp_file.write(sentences.encode())
    tmp_file.close()

    # ======================================================
    # tokenize sentence
    # ======================================================
    cmd.append(osp.basename(tmp_file.name))
    p_tokenizer = subprocess.Popen(
        cmd,
        cwd=path_to_jar_dirname,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL if verbose < 2 else None,
    )
    token_lines = p_tokenizer.communicate(input=sentences.rstrip().encode())[0]
    token_lines = token_lines.decode()
    lines = token_lines.split("\n")
    # remove temp file
    os.remove(tmp_file.name)

    # ======================================================
    # create dictionary for tokenized captions
    # ======================================================
    outs: Any = [None for _ in range(len(lines))]
    if len(audio_ids) != len(lines):
        raise RuntimeError(
            f"PTB tokenize error: expected {len(audio_ids)} lines in output file but found {len(lines)}."
        )

    for k, line in zip(audio_ids, lines):
        tokenized_caption = [
            w for w in line.rstrip().split(" ") if w not in PTB_PUNCTUATIONS
        ]
        outs[k] = tokenized_caption
    assert all(out is not None for out in outs)

    if verbose >= 2:
        duration = time.perf_counter() - start_time
        logging.debug(f"Tokenization finished in {duration:.2f}s.")

    return outs
