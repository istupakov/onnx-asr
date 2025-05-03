"""PyAnnote VAD implementation."""

import typing
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

from onnx_asr.vad import Vad


class PyAnnoteVad(Vad):
    """PyAnnote VAD implementation."""

    def __init__(self, model_files: dict[str, Path], **kwargs: typing.Any):
        """Create PyAnnote VAD.

        Args:
            model_files: Dict with paths to model files.
            kwargs: Additional parameters for onnxruntime.InferenceSession.

        """
        self._model = rt.InferenceSession(model_files["model"], **kwargs)

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        return {"model": f"**/model{suffix}.onnx"}

    def _encode(self, waveforms: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        (logits,) = self._model.run(["logits"], {"input_values": waveforms[:, None]})
        return typing.cast(npt.NDArray[np.float32], logits)
