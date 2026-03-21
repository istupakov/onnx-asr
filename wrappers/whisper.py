"""Wrapper for Whisper models."""

from collections.abc import Iterator
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import whisper

from onnx_asr.asr import TimestampedResult


class WhisperASR:
    """Wrapper model for Whisper ASR."""

    def __init__(self, name: str, device: str = "cpu"):
        """Create wrapper."""
        self.device = device
        self.model: Any = whisper.load_model(name, device)

    @staticmethod
    def _get_sample_rate() -> Literal[8_000, 16_000]:
        return 16_000

    def recognize_batch(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64], /, **kwargs: object | None
    ) -> Iterator[TimestampedResult]:
        """Recognize waveforms batch."""
        language = kwargs.get("language")
        for waveform, waveform_len in zip(waveforms, waveforms_len, strict=True):
            result = self.model.transcribe(waveform[:waveform_len], language=language)
            yield TimestampedResult(result["text"])
