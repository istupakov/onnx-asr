"""Wrapper for GigaAM models."""

from collections.abc import Iterator
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import torch
from transformers import AutoModel

from onnx_asr.asr import TimestampedResult


class GigaamASR:
    """Wrapper model for GigaAM ASR."""

    def __init__(self, revision: str, device: str = "cpu"):
        """Create wrapper."""
        self.device = device
        self.model: Any = AutoModel.from_pretrained(
            "ai-sage/GigaAM-v3",
            revision=revision,
            trust_remote_code=True,
        ).to(self.device)

    @staticmethod
    def _get_sample_rate() -> Literal[8_000, 16_000]:
        return 16_000

    def recognize_batch(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64], /, **kwargs: object | None
    ) -> Iterator[TimestampedResult]:
        """Recognize waveforms batch."""
        results = self.model.model.decoding.decode(
            self.model.model.head,
            *self.model.forward(
                torch.from_numpy(waveforms).to(self.device), torch.from_numpy(waveforms_len).to(self.device)
            ),
        )
        return (TimestampedResult(text) for text in results)
