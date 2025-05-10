"""ASR preprocessor implementations."""

from importlib.resources import files
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

from onnx_asr.utils import OnnxSessionOptions, is_float32_array, is_int64_array


class Preprocessor:
    """ASR preprocessor implementation."""

    def __init__(self, name: str, onnx_options: OnnxSessionOptions):
        """Create ASR preprocessor.

        Args:
            name: Preprocessor name.
            onnx_options: Options for onnxruntime InferenceSession.

        """
        filename = str(Path(name).with_suffix(".onnx"))
        if onnx_options.get("cpu_preprocessing", False):
            onnx_options = {"sess_options": onnx_options.get("sess_options")}
        self._preprocessor = rt.InferenceSession(files(__package__).joinpath(filename).read_bytes(), **onnx_options)

    def __call__(
        self, waveforms: npt.NDArray[np.float32], waveforms_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        """Convert waveforms to model features."""
        features, features_lens = self._preprocessor.run(
            ["features", "features_lens"], {"waveforms": waveforms, "waveforms_lens": waveforms_lens}
        )
        assert is_float32_array(features) and is_int64_array(features_lens)
        return features, features_lens
