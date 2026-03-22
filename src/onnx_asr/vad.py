"""Base VAD classes."""

from abc import abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import chain, islice
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt

from onnx_asr.asr import Asr, TimestampedResult
from onnx_asr.utils import pad_list


@dataclass
class SegmentResult:
    """Segment recognition result."""

    start: float
    """Segment start time."""
    end: float
    """Segment end time."""
    text: str
    """Segment recognized text."""


@dataclass
class TimestampedSegmentResult(TimestampedResult, SegmentResult):
    """Timestamped segment recognition result."""


class Vad(Protocol):
    """VAD protocol."""

    def recognize_batch(
        self,
        asr: Asr,
        waveforms: npt.NDArray[np.float32],
        waveforms_len: npt.NDArray[np.int64],
        sample_rate: Literal[8_000, 16_000],
        asr_kwargs: dict[str, object | None],
        batch_size: int = 8,
        **kwargs: float,
    ) -> Iterator[Iterator[TimestampedSegmentResult]]:
        """Segment and recognize waveforms batch."""
        ...


class BaseVad(Vad):
    """Base VAD class."""

    INF = 10**15

    def _merge_segments(
        self,
        segments: Iterator[tuple[int, int]],
        waveform_len: int,
        sample_rate: int,
        *,
        min_speech_duration_ms: float = 250,
        max_speech_duration_s: float = 20,
        min_silence_duration_ms: float = 100,
        speech_pad_ms: float = 30,
        **kwargs: float,
    ) -> Iterator[tuple[int, int]]:
        speech_pad = int(speech_pad_ms * sample_rate // 1000)
        min_speech_duration = int(min_speech_duration_ms * sample_rate // 1000) - 2 * speech_pad
        max_speech_duration = int(max_speech_duration_s * sample_rate) - 2 * speech_pad
        min_silence_duration = int(min_silence_duration_ms * sample_rate // 1000) + 2 * speech_pad

        cur_start, cur_end = -self.INF, -self.INF
        for start, end in chain(segments, ((waveform_len, waveform_len), (self.INF, self.INF))):
            if start - cur_end < min_silence_duration and end - cur_start < max_speech_duration:
                cur_end = end
            else:
                if cur_end - cur_start > min_speech_duration:
                    yield max(cur_start - speech_pad, 0), min(cur_end + speech_pad, waveform_len)
                while end - start > max_speech_duration:
                    yield max(start - speech_pad, 0), start + max_speech_duration + speech_pad
                    start += max_speech_duration
                cur_start, cur_end = start, end

    @abstractmethod
    def segment_batch(
        self,
        waveforms: npt.NDArray[np.float32],
        waveforms_len: npt.NDArray[np.int64],
        sample_rate: Literal[8_000, 16_000],
        **kwargs: float,
    ) -> Iterator[Iterator[tuple[int, int]]]:
        """Segment waveforms batch."""
        ...

    def recognize_batch(
        self,
        asr: Asr,
        waveforms: npt.NDArray[np.float32],
        waveforms_len: npt.NDArray[np.int64],
        sample_rate: Literal[8_000, 16_000],
        asr_kwargs: dict[str, object | None],
        batch_size: int = 8,
        **kwargs: float,
    ) -> Iterator[Iterator[TimestampedSegmentResult]]:
        """Segment and recognize waveforms batch."""

        def recognize(
            waveform: npt.NDArray[np.float32], segment: Iterator[tuple[int, int]]
        ) -> Iterator[TimestampedSegmentResult]:
            while batch := tuple(islice(segment, int(batch_size))):
                yield from (
                    TimestampedSegmentResult(
                        start / sample_rate, end / sample_rate, res.text, res.timestamps, res.tokens, res.logprobs
                    )
                    for res, (start, end) in zip(
                        asr.recognize_batch(*pad_list([waveform[start:end] for start, end in batch]), **asr_kwargs),
                        batch,
                        strict=True,
                    )
                )

        return map(recognize, waveforms, self.segment_batch(waveforms, waveforms_len, sample_rate, **kwargs))
