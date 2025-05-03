"""Base VAD classes."""

from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt


class Vad(ABC):
    """Abstract VAD class with common interface and methods."""

    SAMPLE_RATE = 16_000

    @abstractmethod
    def _find_segments(self, waveforms: npt.NDArray[np.float32], **kwargs: float) -> Iterable[Iterable[tuple[int, int]]]: ...

    def _process_segments(
        self,
        segments: Iterable[tuple[int, int]],
        max_end: int,
        min_speech_duration: float = 250,
        min_silence_duration: float = 100,
        speech_pad: float = 30,
        **kwargs: float,
    ) -> Iterable[tuple[int, int]]:
        min_speech_duration *= self.SAMPLE_RATE // 1000
        min_silence_duration *= self.SAMPLE_RATE // 1000
        speech_pad = int(self.SAMPLE_RATE * speech_pad // 1000)

        cur_start, cur_end = -self.SAMPLE_RATE, -self.SAMPLE_RATE
        for start, end in segments:
            if start - cur_end < min_silence_duration + 2 * speech_pad:
                cur_end = end
            else:
                if cur_end - cur_start > min_speech_duration - 2 * speech_pad:
                    yield max(cur_start - speech_pad, 0), min(cur_end + speech_pad, max_end)
                cur_start, cur_end = start, end

        if cur_end - cur_start > min_speech_duration:
            yield max(cur_start - speech_pad, 0), min(cur_end + speech_pad, max_end)

    def segment_batch(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64], **kwargs: float
    ) -> Iterable[Iterable[tuple[int, int]]]:
        """Segment waveforms batch."""
        return (
            self._process_segments(segments, max_end, **kwargs)
            for segments, max_end in zip(self._find_segments(waveforms, **kwargs), waveforms_len, strict=True)
        )
