"""Wrapper for GigaAM models."""

from collections.abc import Iterator
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from tone import DecoderType, StreamingCTCPipeline

from onnx_asr.asr import TimestampedResult


class TOneASR:
    """Wrapper model for T-One ASR."""

    def __init__(self):
        """Create wrapper."""
        self.model: Any = StreamingCTCPipeline.from_hugging_face(decoder_type=DecoderType.GREEDY)

    @staticmethod
    def _get_sample_rate() -> Literal[8_000, 16_000]:
        return 8_000

    def recognize_batch(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64], /, **kwargs: object | None
    ) -> Iterator[TimestampedResult]:
        """Recognize waveforms batch."""
        for waveform, waveform_len in zip(waveforms, waveforms_len, strict=True):
            result = self.model.forward_offline(((2**15 - 1) * waveform[:waveform_len]).astype(np.int32))
            yield TimestampedResult(" ".join(r.text for r in result))
