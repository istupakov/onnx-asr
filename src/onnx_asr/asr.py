"""Base ASR classes."""

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, Literal, TypedDict, TypeVar

import numpy as np
import numpy.typing as npt

from onnx_asr.onnx import OnnxSessionOptions
from onnx_asr.preprocessors import Preprocessor, PreprocessorRuntimeConfig
from onnx_asr.utils import log_softmax

S = TypeVar("S")


@dataclass
class TimestampedResult:
    """Timestamped recognition result."""

    text: str
    timestamps: list[float] | None = None
    tokens: list[str] | None = None
    logprobs: list[float] | None = None


class AsrConfig(TypedDict, total=False):
    """Config for ASR model."""

    model_type: str
    features_size: int
    subsampling_factor: int
    max_tokens_per_step: int
    max_sequence_length: int


@dataclass()
class AsrRuntimeConfig:
    """ASR runtime config."""

    onnx_options: OnnxSessionOptions = field(default_factory=OnnxSessionOptions)
    preprocessor_config: PreprocessorRuntimeConfig = field(default_factory=PreprocessorRuntimeConfig)


class Asr(ABC):
    """Base ASR class."""

    def __init__(self, model_files: dict[str, Path], runtime_config: AsrRuntimeConfig):
        """Init base ASR class."""
        if "config" in model_files:
            with model_files["config"].open("rt", encoding="utf-8") as f:
                self.config: AsrConfig = json.load(f)
        else:
            self.config = {}

        self.runtime_config = runtime_config
        self._preprocessor = Preprocessor(self._preprocessor_name, runtime_config.preprocessor_config)

    @staticmethod
    def _get_excluded_providers() -> list[str]:
        return []

    @staticmethod
    def _get_sample_rate() -> Literal[8_000, 16_000]:
        return 16_000

    @property
    @abstractmethod
    def _preprocessor_name(self) -> str: ...

    @abstractmethod
    def recognize_batch(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64], /, **kwargs: str | None
    ) -> Iterator[TimestampedResult]:
        """Recognize waveforms batch."""
        ...


class _AsrWithDecoding(Asr):
    DECODE_SPACE_PATTERN = re.compile(r"\A\s|\s\B|(\s)\b")
    window_step = 0.01

    def __init__(self, model_files: dict[str, Path], runtime_config: AsrRuntimeConfig):
        super().__init__(model_files, runtime_config)

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
        self, encoder_out: npt.NDArray[np.float32], encoder_out_lens: npt.NDArray[np.int64], /, **kwargs: str | None
    ) -> Iterator[tuple[Iterable[int], Iterable[int] | None, Iterable[float] | None]]: ...

    def _decode_tokens(
        self, ids: Iterable[int], indices: Iterable[int] | None, logprobs: Iterable[float] | None
    ) -> TimestampedResult:
        tokens = [self._vocab[i] for i in ids]
        text = re.sub(self.DECODE_SPACE_PATTERN, lambda x: " " if x.group(1) else "", "".join(tokens))
        timestamps = None if indices is None else (self.window_step * self._subsampling_factor * np.asarray(indices)).tolist()
        return TimestampedResult(text, timestamps, tokens, None if logprobs is None else np.asarray(logprobs).tolist())

    def recognize_batch(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64], /, **kwargs: str | None
    ) -> Iterator[TimestampedResult]:
        encoder_out, encoder_out_lens = self._encode(*self._preprocessor(waveforms, waveforms_len))
        return map(self._decode_tokens, *zip(*self._decoding(encoder_out, encoder_out_lens, **kwargs), strict=False))


class _AsrWithCtcDecoding(_AsrWithDecoding):
    def _decoding(
        self, encoder_out: npt.NDArray[np.float32], encoder_out_lens: npt.NDArray[np.int64], /, **kwargs: str | None
    ) -> Iterator[tuple[Iterable[int], Iterable[int], Iterable[float]]]:
        assert encoder_out.shape[-1] <= self._vocab_size
        assert encoder_out.shape[1] >= max(encoder_out_lens)

        batch_tokens = encoder_out.argmax(axis=-1)
        batch_mask = np.arange(batch_tokens.shape[1])[None, :] < encoder_out_lens[:, None]
        batch_mask &= batch_tokens != self._blank_idx
        batch_logprobs = np.take_along_axis(encoder_out, batch_tokens[:, :, None], axis=-1).squeeze(axis=-1)
        batch_logprobs = np.where(batch_mask, batch_logprobs, 0.0)
        batch_mask &= np.diff(batch_tokens, axis=-1, prepend=self._blank_idx) != 0
        for i in range(encoder_out.shape[0]):
            mask = batch_mask[i]
            idx = np.flatnonzero(mask)
            yield batch_tokens[i][mask], idx, np.add.reduceat(batch_logprobs[i], idx)


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
        self, encoder_out: npt.NDArray[np.float32], encoder_out_lens: npt.NDArray[np.int64], /, **kwargs: str | None
    ) -> Iterator[tuple[Iterable[int], Iterable[int], Iterable[float] | None]]:
        need_logprobs = kwargs.get("need_logprobs")
        for encodings, encodings_len in zip(encoder_out, encoder_out_lens, strict=True):
            assert encodings_len <= encodings.shape[0]
            prev_state = self._create_state()
            tokens: list[int] = []
            timestamps: list[int] = []
            logprobs: list[float] = []

            t = 0
            emitted_tokens = 0
            while t < encodings_len:
                logits, step, state = self._decode(tokens, prev_state, encodings[t])
                assert logits.shape[-1] <= self._vocab_size

                token = logits.argmax()

                if token != self._blank_idx:
                    prev_state = state
                    tokens.append(int(token))
                    timestamps.append(t)
                    emitted_tokens += 1
                    if need_logprobs:
                        logprobs.append(log_softmax(logits)[token])

                if step > 0:
                    t += step
                    emitted_tokens = 0
                elif token == self._blank_idx or emitted_tokens == self._max_tokens_per_step:
                    t += 1
                    emitted_tokens = 0

            yield tokens, timestamps, logprobs if need_logprobs else None
