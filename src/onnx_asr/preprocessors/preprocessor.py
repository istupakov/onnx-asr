"""ASR preprocessor implementations."""

from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

from onnx_asr.utils import OnnxSessionOptions, is_float32_array, is_int64_array


@dataclass()
class PreprocessorRuntimeConfig:
    """Preprocessor runtime config."""

    onnx_options: OnnxSessionOptions = field(default_factory=OnnxSessionOptions)


class Preprocessor:
    """ASR preprocessor implementation."""

    def __init__(self, name: str, runtime_config: PreprocessorRuntimeConfig):
        """Create ASR preprocessor.

        Args:
            name: Preprocessor name.
            runtime_config: Runtime configuration.

        """
        if name == "identity":
            self._preprocessor = None
            return

        filename = str(Path(name).with_suffix(".onnx"))
        self._preprocessor = rt.InferenceSession(
            files(__package__).joinpath(filename).read_bytes(), **runtime_config.onnx_options
        )

    def __call__(
        self, waveforms: npt.NDArray[np.float32], waveforms_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        """Convert waveforms to model features."""
        if not self._preprocessor:
            return waveforms, waveforms_lens

        features, features_lens = self._preprocessor.run(
            ["features", "features_lens"], {"waveforms": waveforms, "waveforms_lens": waveforms_lens}
        )
        assert is_float32_array(features)
        assert is_int64_array(features_lens)
        return features, features_lens
