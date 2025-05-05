"""Base VAD classes."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import islice

import numpy as np
import numpy.typing as npt

from .asr import Asr, TimestampedResult
from .utils import pad_list


@dataclass
class SegmentResult:
    """Segment recognition result."""

    start: float
    end: float
    text: str


@dataclass
class TimestampedSegmentResult(TimestampedResult, SegmentResult):
    """Timestamped segment recognition result."""


class Vad(ABC):
    """Base VAD class."""

    SAMPLE_RATE = 16_000

    @abstractmethod
    def segment_batch(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64], **kwargs: float
    ) -> Iterator[Iterator[tuple[int, int]]]:
        """Segment waveforms batch."""
        ...

    def recognize_batch(
        self,
        asr: Asr,
        waveforms: npt.NDArray[np.float32],
        waveforms_len: npt.NDArray[np.int64],
        language: str | None,
        batch_size: float = 8,
        **kwargs: float,
    ) -> Iterator[Iterator[TimestampedSegmentResult]]:
        """Segment and recognize waveforms batch."""

        def recognize(
            waveform: npt.NDArray[np.float32], segment: Iterator[tuple[int, int]]
        ) -> Iterator[TimestampedSegmentResult]:
            while batch := tuple(islice(segment, int(batch_size))):
                yield from (
                    TimestampedSegmentResult(
                        start / self.SAMPLE_RATE, end / self.SAMPLE_RATE, res.text, res.timestamps, res.tokens
                    )
                    for res, (start, end) in zip(
                        asr.recognize_batch(*pad_list([waveform[start:end] for start, end in batch]), language),
                        batch,
                        strict=True,
                    )
                )

        return map(recognize, waveforms, self.segment_batch(waveforms, waveforms_len, **kwargs))
