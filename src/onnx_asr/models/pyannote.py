"""PyAnnote VAD implementation."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

from onnx_asr.onnx import OnnxSessionOptions
from onnx_asr.utils import is_float32_array
from onnx_asr.vad import Vad


class PyAnnoteVad(Vad):
    """PyAnnote VAD implementation."""

    def __init__(self, model_files: dict[str, Path], onnx_options: OnnxSessionOptions):
        """Create PyAnnote VAD.

        Args:
            model_files: Dict with paths to model files.
            onnx_options: Options for onnxruntime InferenceSession.

        """
        self._model = rt.InferenceSession(model_files["model"], **onnx_options)

    @staticmethod
    def _get_excluded_providers() -> list[str]:
        return []

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        return {"model": f"**/model{suffix}.onnx"}

    def _encode(self, waveforms: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        (logits,) = self._model.run(["logits"], {"input_values": waveforms[:, None]})
        assert is_float32_array(logits)
        return logits
