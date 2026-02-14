"""Wrapper for sherpa-onnx models."""

from collections.abc import Iterator
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from onnx_asr.asr import TimestampedResult


class NemoASR:
    """Wrapper model for NeMo Toolkit ASR."""

    def __init__(self, model_name: str):
        """Create wrapper."""
        from nemo.utils.nemo_logging import Logger

        self.logger = Logger()
        self.logger.setLevel(Logger.ERROR)

        import nemo.collections.asr as nemo_asr

        self.model: Any = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
        self.model.change_decoding_strategy({"strategy": "greedy_batch"})
        self.model.eval()

    @staticmethod
    def _get_sample_rate() -> Literal[8_000, 16_000]:
        return 16_000

    def recognize_batch(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64], /, **kwargs: object | None
    ) -> Iterator[TimestampedResult]:
        """Recognize waveforms batch."""
        for waveform, waveform_len in zip(waveforms, waveforms_len, strict=True):
            hypot = self.model.transcribe(waveform[:waveform_len], verbose=False)
            yield TimestampedResult(hypot[0].text)
