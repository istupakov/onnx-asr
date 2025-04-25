"""ASR preprocessor implementations."""

from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import onnxruntime as rt


class Preprocessor:
    """ASR preprocessor implementation."""

    def __init__(self, name: str, **kwargs: Any):
        """Create ASR preprocessor.

        Args:
            name: Preprocessor name.
            kwargs: Additional parameters for onnxruntime.InferenceSession.

        """
        filename = str(Path(name).with_suffix(".onnx"))
        self._preprocessor = rt.InferenceSession(files(__package__).joinpath(filename).read_bytes(), **kwargs)

    def __call__(
        self, waveforms: npt.NDArray[np.float32], waveforms_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        """Convert waveforms to model features."""
        features, features_lens = self._preprocessor.run(
            ["features", "features_lens"], {"waveforms": waveforms, "waveforms_lens": waveforms_lens}
        )
        return features, features_lens
