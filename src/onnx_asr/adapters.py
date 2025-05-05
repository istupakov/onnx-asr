"""ASR adapter classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Generic, TypeVar, overload

import numpy as np
import numpy.typing as npt

from .asr import Asr, TimestampedResult
from .preprocessors import Resampler
from .utils import SampleRates, read_wav_files
from .vad import SegmentResult, TimestampedSegmentResult, Vad

R = TypeVar("R")


class AsrAdapter(ABC, Generic[R]):
    """Base ASR adapter class."""

    asr: Asr
    resampler: Resampler

    def __init__(self, asr: Asr, resampler: Resampler):
        """Create ASR adapter."""
        self.asr = asr
        self.resampler = resampler

    def with_vad(self, vad: Vad, **kwargs: float) -> SegmentResultsAsrAdapter:
        """ASR with VAD adapter (text results)."""
        return SegmentResultsAsrAdapter(self.asr, vad, self.resampler, **kwargs)

    @abstractmethod
    def _recognize_batch(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64], language: str | None
    ) -> Iterator[R]: ...

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
            return list(self._recognize_batch(*self.resampler(*read_wav_files(waveform, sample_rate)), language))
        return next(self._recognize_batch(*self.resampler(*read_wav_files([waveform], sample_rate)), language))


class TimestampedResultsAsrAdapter(AsrAdapter[TimestampedResult]):
    """ASR adapter (timestamped results)."""

    def _recognize_batch(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64], language: str | None
    ) -> Iterator[TimestampedResult]:
        return self.asr.recognize_batch(waveforms, waveforms_len, language)


class TextResultsAsrAdapter(AsrAdapter[str]):
    """ASR adapter (text results)."""

    def with_timestamps(self) -> TimestampedResultsAsrAdapter:
        """ASR adapter (timestamped results)."""
        return TimestampedResultsAsrAdapter(self.asr, self.resampler)

    def _recognize_batch(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64], language: str | None
    ) -> Iterator[str]:
        return (res.text for res in self.asr.recognize_batch(waveforms, waveforms_len, language))


class TimestampedSegmentResultsAsrAdapter(AsrAdapter[Iterator[TimestampedSegmentResult]]):
    """ASR with VAD adapter (timestamped results)."""

    vad: Vad

    def __init__(self, asr: Asr, vad: Vad, resampler: Resampler, **kwargs: float):
        """Create ASR adapter."""
        super().__init__(asr, resampler)
        self.vad = vad
        self._vadargs = kwargs

    def _recognize_batch(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64], language: str | None
    ) -> Iterator[Iterator[TimestampedSegmentResult]]:
        return self.vad.recognize_batch(self.asr, waveforms, waveforms_len, language, **self._vadargs)


class SegmentResultsAsrAdapter(AsrAdapter[Iterator[SegmentResult]]):
    """ASR with VAD adapter (text results)."""

    vad: Vad

    def __init__(self, asr: Asr, vad: Vad, resampler: Resampler, **kwargs: float):
        """Create ASR adapter."""
        super().__init__(asr, resampler)
        self.vad = vad
        self._vadargs = kwargs

    def with_timestamps(self) -> TimestampedSegmentResultsAsrAdapter:
        """ASR with VAD adapter (timestamped results)."""
        return TimestampedSegmentResultsAsrAdapter(self.asr, self.vad, self.resampler, **self._vadargs)

    def _recognize_batch(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64], language: str | None
    ) -> Iterator[Iterator[SegmentResult]]:
        return (
            (SegmentResult(res.start, res.end, res.text) for res in results)
            for results in self.vad.recognize_batch(self.asr, waveforms, waveforms_len, language, **self._vadargs)
        )
