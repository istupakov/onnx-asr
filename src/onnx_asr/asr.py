"""Base ASR classes."""

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .preprocessors import Preprocessor, Resampler
from .utils import SampleRates, pad_list, read_wav_files


@dataclass
class Result:
    """ASR recognition result."""

    text: str
    timestamps: list[float] | None = None
    tokens: list[str] | None = None


class Asr(ABC):
    """Abstract ASR class with common interface and methods."""

    def __init__(self, **kwargs: Any):
        """Create ASR."""
        self._resampler = Resampler(**kwargs)

    @abstractmethod
    def _recognize_batch(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64], language: str | None
    ) -> list[Result]: ...

    def _recognize(
        self, waveforms: list[str | npt.NDArray[np.float32]], sample_rate: SampleRates, language: str | None, only_text: bool
    ) -> list[str] | list[Result]:
        waveform_arrays, sample_rate = read_wav_files(waveforms, sample_rate)
        results = self._recognize_batch(*self._resampler(*pad_list(waveform_arrays), sample_rate), language)
        if only_text:
            return [res.text for res in results]
        return results

    def recognize(
        self,
        waveform: str | npt.NDArray[np.float32] | list[str | npt.NDArray[np.float32]],
        *,
        sample_rate: SampleRates = 16_000,
        language: str | None = None,
        only_text: bool = True,
    ) -> str | Result | list[str] | list[Result]:
        """Recognize speech (single or batch).

        Args:
            waveform: Path to wav file (only PCM_U8, PCM_16, PCM_24 and PCM_32 formats are supported)
                      or Numpy array with PCM waveform.
                      A list of file paths or numpy arrays for batch recognition are also supported.
            sample_rate: Sample rate for Numpy arrays in waveform.
            language: Speech language (only for Whisper models).
            only_text: Return only texts or Result objects

        Returns:
            Speech recognition results (single result or list for batch recognition).

        """
        if isinstance(waveform, list):
            if not waveform:
                return []
            return self._recognize(waveform, sample_rate, language, only_text)
        return self._recognize([waveform], sample_rate, language, only_text)[0]


class _AsrWithDecoding(Asr):
    DECODE_SPACE_PATTERN = re.compile(r"\A\s|\s\B|(\s)\b")

    def __init__(self, preprocessor_name: str, vocab_path: Path, **kwargs: Any):
        super().__init__(**kwargs)
        self._preprocessor = Preprocessor(preprocessor_name, **kwargs)
        with Path(vocab_path).open("rt", encoding="utf-8") as f:
            tokens = {token: int(id) for token, id in (line.strip("\n").split(" ") for line in f.readlines())}
        self._vocab = {id: token.replace("\u2581", " ") for token, id in tokens.items()}
        self._blank_idx = tokens["<blk>"]

    @abstractmethod
    def _encode(
        self, features: npt.NDArray[np.float32], features_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]: ...

    @abstractmethod
    def _decoding(
        self, encoder_out: npt.NDArray[np.float32], encoder_out_lens: npt.NDArray[np.int64]
    ) -> Iterable[tuple[list[int], list[int]]]: ...

    def _decode_tokens(self, ids: list[int], timestamps: list[float]) -> Result:
        tokens = [self._vocab[i] for i in ids]
        text = re.sub(self.DECODE_SPACE_PATTERN, lambda x: " " if x.group(1) else "", "".join(tokens))
        return Result(text, timestamps, tokens)

    def _recognize_batch(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64], language: str | None
    ) -> list[Result]:
        encoder_out, encoder_out_lens = self._encode(*self._preprocessor(waveforms, waveforms_len))
        subsampling = np.round((waveforms_len / encoder_out_lens / 160).mean())
        return [
            self._decode_tokens(tokens, (0.01 * subsampling * np.array(timestamps)).tolist())
            for tokens, timestamps in self._decoding(encoder_out, encoder_out_lens)
        ]


class _AsrWithCtcDecoding(_AsrWithDecoding):
    def _decoding(
        self, encoder_out: npt.NDArray[np.float32], encoder_out_lens: npt.NDArray[np.int64]
    ) -> Iterable[tuple[list[int], list[int]]]:
        assert encoder_out.shape[-1] <= len(self._vocab)

        for log_probs, log_probs_len in zip(encoder_out, encoder_out_lens, strict=True):
            tokens = log_probs[:log_probs_len].argmax(axis=-1)
            indices = np.flatnonzero(np.diff(tokens))
            tokens = tokens[indices]
            mask = tokens != self._blank_idx
            yield tokens[mask].tolist(), indices[mask].tolist()


class _AsrWithRnntDecoding(_AsrWithDecoding):
    @abstractmethod
    def _create_state(self) -> tuple: ...

    @property
    @abstractmethod
    def _max_tokens_per_step(self) -> int: ...

    @abstractmethod
    def _decode(
        self, prev_tokens: list[int], prev_state: tuple, encoder_out: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], tuple]: ...

    def _decoding(
        self, encoder_out: npt.NDArray[np.float32], encoder_out_lens: npt.NDArray[np.int64]
    ) -> Iterable[tuple[list[int], list[int]]]:
        for encodings, encodings_len in zip(encoder_out, encoder_out_lens, strict=True):
            prev_state = self._create_state()
            tokens: list[int] = []
            timestamps: list[int] = []

            for t in range(encodings_len):
                emitted_tokens = 0
                while emitted_tokens < self._max_tokens_per_step:
                    probs, state = self._decode(tokens, prev_state, encodings[:, t])
                    assert probs.shape[-1] <= len(self._vocab)

                    token = probs.argmax()

                    if token != self._blank_idx:
                        prev_state = state
                        tokens.append(int(token))
                        timestamps.append(t)
                        emitted_tokens += 1
                    else:
                        break

            yield tokens, timestamps
