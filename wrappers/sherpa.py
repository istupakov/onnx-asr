"""Wrapper for sherpa-onnx models."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from onnx_asr.asr import TimestampedResult
from onnx_asr.resolver import Resolver


class SherpaASR:
    """Wrapper model for sherpa-onnx ASR."""

    def __init__(
        self,
        repo_id: str | None = None,
        local_dir: Path | None = None,
        *,
        offline: bool | None = None,
        quantization: str | None = None,
        **kwargs: Any,
    ):
        """Create wrapper."""
        resolver = Resolver(SherpaASR, repo_id, local_dir, offline=offline)
        model_files = resolver.resolve_model(quantization=quantization)

        import sherpa_onnx

        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            str(model_files["encoder"]),
            str(model_files["decoder"]),
            str(model_files["joiner"]),
            str(model_files["tokens"]),
            bpe_vocab=str(model_files["bpe_vocab"]),
            modeling_unit="bpe",
            **kwargs,
        )

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        return {
            "encoder": f"*/encoder{suffix}.onnx",
            "decoder": f"*/decoder{suffix}.onnx",
            "joiner": f"*/joiner{suffix}.onnx",
            "tokens": "*/tokens.txt",
            "bpe_vocab": "*/unigram_500.vocab",
        }

    @staticmethod
    def _get_sample_rate() -> Literal[8_000, 16_000]:
        return 16_000

    def recognize_batch(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64], /, **kwargs: object | None
    ) -> Iterator[TimestampedResult]:
        """Recognize waveforms batch."""
        streams = []
        for waveform, waveform_len in zip(waveforms, waveforms_len, strict=True):
            stream = self._recognizer.create_stream()
            stream.accept_waveform(self._get_sample_rate(), waveform[:waveform_len])
            streams.append(stream)
        self._recognizer.decode_streams(streams)
        return (
            TimestampedResult(
                stream.result.text, stream.result.timestamps, stream.result.tokens, stream.result.ys_log_probs
            )
            for stream in streams
        )
