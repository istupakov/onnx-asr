"""Wespeaker SE implementation."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

from onnx_asr.asr import Preprocessor
from onnx_asr.onnx import OnnxSessionOptions
from onnx_asr.se import SpeakerEmbedding
from onnx_asr.utils import is_float32_array


class WespeakerEmbeddings(SpeakerEmbedding):
    """Wespeaker embeddings model."""

    def __init__(
        self,
        model_files: dict[str, Path],
        preprocessor_factory: Callable[[str], Preprocessor],
        onnx_options: OnnxSessionOptions,
    ):
        """Create model.

        Args:
            model_files: Dict with paths to model files.
            preprocessor_factory: Factory for preprocessor creation.
            onnx_options: Options for onnxruntime InferenceSession.

        """
        self._model = rt.InferenceSession(model_files["model"], **onnx_options)
        self._preprocessor = preprocessor_factory("wespeaker")

    @staticmethod
    def _get_excluded_providers() -> list[str]:
        return []

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        return {"config": "config.yaml", "model": f"*{suffix}.onnx"}

    def embedding(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float32]:
        """Compute speaker embedding."""
        features, _ = self._preprocessor(waveforms, waveforms_len)
        features -= features.mean(axis=1, keepdims=True)
        (embs,) = self._model.run(["embs"], {"feats": features})
        assert is_float32_array(embs)
        return embs
