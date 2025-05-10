"""Base ASR classes."""

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypedDict, TypeVar

import numpy as np
import numpy.typing as npt

from .preprocessors import Preprocessor
from .utils import OnnxSessionOptions

S = TypeVar("S")


@dataclass
class TimestampedResult:
    """Timestamped recognition result."""

    text: str
    timestamps: list[float] | None = None
    tokens: list[str] | None = None


class AsrConfig(TypedDict, total=False):
    """Config for ASR model."""

    model_type: str
    features_size: int
    subsampling_factor: int
    max_tokens_per_step: int


class Asr(ABC):
    """Base ASR class."""

    def __init__(self, model_files: dict[str, Path], onnx_options: OnnxSessionOptions):
        """Init base ASR class."""
        if "config" in model_files:
            with model_files["config"].open("rt", encoding="utf-8") as f:
                self.config: AsrConfig = json.load(f)
        else:
            self.config = {}

        self._preprocessor = Preprocessor(self._preprocessor_name, onnx_options)

    @property
    @abstractmethod
    def _preprocessor_name(self) -> str: ...

    @abstractmethod
    def recognize_batch(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64], language: str | None
    ) -> Iterator[TimestampedResult]:
        """Recognize waveforms batch."""
        ...


class _AsrWithDecoding(Asr):
    DECODE_SPACE_PATTERN = re.compile(r"\A\s|\s\B|(\s)\b")
    window_size = 0.01

    def __init__(self, model_files: dict[str, Path], onnx_options: OnnxSessionOptions):
        super().__init__(model_files, onnx_options)

        with Path(model_files["vocab"]).open("rt", encoding="utf-8") as f:
            tokens = {token: int(id) for token, id in (line.strip("\n").split(" ") for line in f.readlines())}
        self._vocab = {id: token.replace("\u2581", " ") for token, id in tokens.items()}
        self._vocab_size = len(self._vocab)
        self._blank_idx = tokens["<blk>"]

    @property
    @abstractmethod
    def _subsampling_factor(self) -> int: ...

    @abstractmethod
    def _encode(
        self, features: npt.NDArray[np.float32], features_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]: ...

    @abstractmethod
    def _decoding(
        self, encoder_out: npt.NDArray[np.float32], encoder_out_lens: npt.NDArray[np.int64]
    ) -> Iterator[tuple[list[int], list[int]]]: ...

    def _decode_tokens(self, ids: list[int], timestamps: list[float]) -> TimestampedResult:
        tokens = [self._vocab[i] for i in ids]
        text = re.sub(self.DECODE_SPACE_PATTERN, lambda x: " " if x.group(1) else "", "".join(tokens))
        return TimestampedResult(text, timestamps, tokens)

    def recognize_batch(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64], language: str | None
    ) -> Iterator[TimestampedResult]:
        encoder_out, encoder_out_lens = self._encode(*self._preprocessor(waveforms, waveforms_len))
        return (
            self._decode_tokens(tokens, (self.window_size * self._subsampling_factor * np.array(timestamps)).tolist())
            for tokens, timestamps in self._decoding(encoder_out, encoder_out_lens)
        )


class _AsrWithCtcDecoding(_AsrWithDecoding):
    def _decoding(
        self, encoder_out: npt.NDArray[np.float32], encoder_out_lens: npt.NDArray[np.int64]
    ) -> Iterator[tuple[list[int], list[int]]]:
        assert encoder_out.shape[-1] <= self._vocab_size
        assert encoder_out.shape[1] == max(encoder_out_lens)

        for log_probs, log_probs_len in zip(encoder_out, encoder_out_lens, strict=True):
            tokens = log_probs[:log_probs_len].argmax(axis=-1)
            indices = np.flatnonzero(np.diff(tokens))
            tokens = tokens[indices]
            mask = tokens != self._blank_idx
            yield tokens[mask].tolist(), indices[mask].tolist()


class _AsrWithTransducerDecoding(_AsrWithDecoding, Generic[S]):
    @property
    @abstractmethod
    def _max_tokens_per_step(self) -> int: ...

    @abstractmethod
    def _create_state(self) -> S: ...

    @abstractmethod
    def _decode(
        self, prev_tokens: list[int], prev_state: S, encoder_out: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], int, S]: ...

    def _decoding(
        self, encoder_out: npt.NDArray[np.float32], encoder_out_lens: npt.NDArray[np.int64]
    ) -> Iterator[tuple[list[int], list[int]]]:
        for encodings, encodings_len in zip(encoder_out, encoder_out_lens, strict=True):
            prev_state = self._create_state()
            tokens: list[int] = []
            timestamps: list[int] = []

            t = 0
            emitted_tokens = 0
            while t < encodings_len:
                probs, step, state = self._decode(tokens, prev_state, encodings[:, t])
                assert probs.shape[-1] <= self._vocab_size

                token = probs.argmax()

                if token != self._blank_idx:
                    prev_state = state
                    tokens.append(int(token))
                    timestamps.append(t)
                    emitted_tokens += 1

                if step >= 0:
                    t += step
                    emitted_tokens = 0
                elif token == self._blank_idx or emitted_tokens == self._max_tokens_per_step:
                    t += 1
                    emitted_tokens = 0

            yield tokens, timestamps
