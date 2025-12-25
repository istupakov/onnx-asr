"""Base VAD classes."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import islice
from typing import Literal

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
        asr_kwargs: dict[str, str | None],
        batch_size: float = 8,
        **kwargs: float,
    ) -> Iterator[Iterator[TimestampedSegmentResult]]:
        """Segment and recognize waveforms batch."""

        def recognize(
            waveform: npt.NDArray[np.float32], segment: Iterator[tuple[int, int]]
        ) -> Iterator[TimestampedSegmentResult]:
            while batch := tuple(islice(segment, int(batch_size))):
                yield from (
                    TimestampedSegmentResult(start / sample_rate, end / sample_rate, res.text, res.timestamps, res.tokens)
                    for res, (start, end) in zip(
                        asr.recognize_batch(*pad_list([waveform[start:end] for start, end in batch]), **asr_kwargs),
                        batch,
                        strict=True,
                    )
                )

        return map(recognize, waveforms, self.segment_batch(waveforms, waveforms_len, sample_rate, **kwargs))
