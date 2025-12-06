"""Base ASR classes."""

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Literal, TypedDict, TypeVar

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
    max_sequence_length: int


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

    @staticmethod
    def _get_sample_rate() -> Literal[8_000, 16_000]:
        return 16_000

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
    window_step = 0.01

    def __init__(self, model_files: dict[str, Path], onnx_options: OnnxSessionOptions):
        super().__init__(model_files, onnx_options)

        if "vocab" in model_files:
            with Path(model_files["vocab"]).open("rt", encoding="utf-8") as f:
                self._vocab = {int(id): token.replace("\u2581", " ") for token, id in (line.strip("\n").split(" ") for line in f)}
            self._vocab_size = len(self._vocab)
            if (blank_idx := next((id for id, token in self._vocab.items() if token == "<blk>"), None)) is not None:
                self._blank_idx = blank_idx

    @property
    @abstractmethod
    def _subsampling_factor(self) -> int: ...

    @abstractmethod
    def _encode(
        self, features: npt.NDArray[np.float32], features_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]: ...

    @abstractmethod
    def _decoding(
        self, encoder_out: npt.NDArray[np.float32], encoder_out_lens: npt.NDArray[np.int64], language: str | None
    ) -> Iterator[tuple[Iterable[int], Iterable[int]]]: ...

    def _decode_tokens(self, ids: Iterable[int], indices: Iterable[int]) -> TimestampedResult:
        tokens = [self._vocab[i] for i in ids]
        timestamps = (self.window_step * self._subsampling_factor * np.asarray(indices)).tolist()
        text = re.sub(self.DECODE_SPACE_PATTERN, lambda x: " " if x.group(1) else "", "".join(tokens))
        return TimestampedResult(text, timestamps, tokens)

    def recognize_batch(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64], language: str | None
    ) -> Iterator[TimestampedResult]:
        encoder_out, encoder_out_lens = self._encode(*self._preprocessor(waveforms, waveforms_len))
        return (
            self._decode_tokens(tokens, indices) for tokens, indices in self._decoding(encoder_out, encoder_out_lens, language)
        )


class _AsrWithCtcDecoding(_AsrWithDecoding):
    def _decoding(
        self, encoder_out: npt.NDArray[np.float32], encoder_out_lens: npt.NDArray[np.int64], language: str | None
    ) -> Iterator[tuple[Iterable[int], Iterable[int]]]:
        assert encoder_out.shape[-1] <= self._vocab_size
        assert encoder_out.shape[1] >= max(encoder_out_lens)

        batch_tokens = encoder_out.argmax(axis=-1)
        batch_mask = np.diff(batch_tokens, axis=-1, append=self._blank_idx) != 0
        batch_mask &= batch_tokens != self._blank_idx
        batch_indices = (mask[:mask_len].nonzero() for mask, mask_len in zip(batch_mask, encoder_out_lens, strict=True))
        return ((tokens[indices], indices[0]) for tokens, indices in zip(batch_tokens, batch_indices, strict=True))


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
        self, encoder_out: npt.NDArray[np.float32], encoder_out_lens: npt.NDArray[np.int64], language: str | None
    ) -> Iterator[tuple[Iterable[int], Iterable[int]]]:
        for encodings, encodings_len in zip(encoder_out, encoder_out_lens, strict=True):
            prev_state = self._create_state()
            tokens: list[int] = []
            timestamps: list[int] = []

            t = 0
            emitted_tokens = 0
            while t < encodings_len:
                probs, step, state = self._decode(tokens, prev_state, encodings[t])
                assert probs.shape[-1] <= self._vocab_size

                token = probs.argmax()

                if token != self._blank_idx:
                    prev_state = state
                    tokens.append(int(token))
                    timestamps.append(t)
                    emitted_tokens += 1

                if step > 0:
                    t += step
                    emitted_tokens = 0
                elif token == self._blank_idx or emitted_tokens == self._max_tokens_per_step:
                    t += 1
                    emitted_tokens = 0

            yield tokens, timestamps
