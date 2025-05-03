"""ASR adapter classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar, overload

import numpy as np
import numpy.typing as npt

from .asr import Asr, Result
from .preprocessors import Resampler
from .utils import SampleRates, pad_list, read_wav_files
from .vad import Vad

R = TypeVar("R")


@dataclass
class SegmentResult(Result):
    """Segment recognition result."""

    start: float
    end: float


class AsrAdapter(ABC, Generic[R]):
    """Abstract ASR adapter class with common interface and methods."""

    asr: Asr
    resampler: Resampler

    def __init__(self, asr: Asr, resampler: Resampler):
        """Create ASR adapter."""
        self.asr = asr
        self.resampler = resampler

    def with_vad(self, vad: Vad, **kwargs: float) -> AsrWithVad:
        """ASR with VAD adapter."""
        return AsrWithVad(self.asr, vad, self.resampler, **kwargs)

    @abstractmethod
    def _recognize_batch(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64], language: str | None
    ) -> Iterable[R]: ...

    def _recognize_batch_with_prepare(
        self,
        waveforms: list[str | npt.NDArray[np.float32]],
        sample_rate: SampleRates = 16_000,
        language: str | None = None,
    ) -> Iterable[R]:
        return self._recognize_batch(*self.resampler(*read_wav_files(waveforms, sample_rate)), language)

    @overload
    def recognize(
        self,
        waveform: str | npt.NDArray[np.float32],
        *,
        sample_rate: SampleRates = 16_000,
        language: str | None = None,
    ) -> R: ...

    @overload
    def recognize(
        self,
        waveform: list[str | npt.NDArray[np.float32]],
        *,
        sample_rate: SampleRates = 16_000,
        language: str | None = None,
    ) -> list[R]: ...

    def recognize(
        self,
        waveform: str | npt.NDArray[np.float32] | list[str | npt.NDArray[np.float32]],
        *,
        sample_rate: SampleRates = 16_000,
        language: str | None = None,
    ) -> R | list[R]:
        """Recognize speech (single or batch).

        Args:
            waveform: Path to wav file (only PCM_U8, PCM_16, PCM_24 and PCM_32 formats are supported)
                      or Numpy array with PCM waveform.
                      A list of file paths or numpy arrays for batch recognition are also supported.
            sample_rate: Sample rate for Numpy arrays in waveform.
            language: Speech language (only for Whisper models).

        Returns:
            Speech recognition results (single or list for batch recognition).

        """
        if isinstance(waveform, list):
            if not waveform:
                return []
            return list(self._recognize_batch_with_prepare(waveform, sample_rate, language))
        return next(iter(self._recognize_batch_with_prepare([waveform], sample_rate, language)))


class AsrWithTimestamps(AsrAdapter[Result]):
    """ASR adapter that returns results with timestamps."""

    def without_timestamps(self) -> AsrWithoutTimestamps:
        """ASR adapter that returns text only."""
        return AsrWithoutTimestamps(self.asr, self.resampler)

    def _recognize_batch(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64], language: str | None
    ) -> Iterable[Result]:
        return self.asr.recognize_batch(waveforms, waveforms_len, language)


class AsrWithoutTimestamps(AsrAdapter[str]):
    """ASR adapter that returns text only."""

    def with_timestamps(self) -> AsrWithTimestamps:
        """ASR adapter that returns results with timestamps."""
        return AsrWithTimestamps(self.asr, self.resampler)

    def _recognize_batch(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64], language: str | None
    ) -> Iterable[str]:
        return (res.text for res in self.asr.recognize_batch(waveforms, waveforms_len, language))


class AsrWithVad(AsrAdapter[list[SegmentResult]]):
    """ASR with VAD adapter."""

    vad: Vad

    def __init__(self, asr: Asr, vad: Vad, resampler: Resampler, **kwargs: float):
        """Create ASR with VAD adapter."""
        super().__init__(asr, resampler)
        self.vad = vad
        self._vadargs = kwargs

    def _recognize_segment(
        self, waveform: npt.NDArray[np.float32], segment: list[tuple[int, int]], language: str | None
    ) -> list[SegmentResult]:
        chunks = [waveform[start:end] for start, end in segment]
        return [
            SegmentResult(result.text, result.timestamps, result.tokens, start / self.vad.SAMPLE_RATE, end / self.vad.SAMPLE_RATE)
            for result, (start, end) in zip(self.asr.recognize_batch(*pad_list(chunks), language), segment, strict=True)
        ]

    def _recognize_batch(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64], language: str | None
    ) -> Iterable[list[SegmentResult]]:
        return (
            self._recognize_segment(waveform, list(segment), language)
            for waveform, segment in zip(
                waveforms, self.vad.segment_batch(waveforms, waveforms_len, **self._vadargs), strict=True
            )
        )
