"""ASR preprocessor implementations."""

from importlib.resources import files
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import onnxruntime as rt


class Preprocessor:
    """ASR preprocessor implementation."""

    PreprocessorNames = Literal["gigaam", "kaldi", "nemo", "whisper80", "whisper128"]

    def __init__(self, name: PreprocessorNames, **kwargs):
        """Create ASR preprocessor.

        Args:
            name: Preprocessor name.
            kwargs: Additional parameters for onnxruntime.InferenceSession.

        """
        self._preprocessor = rt.InferenceSession(files(__package__).joinpath(Path(name).with_suffix(".onnx")), **kwargs)  # type: ignore

    def __call__(
        self, waveforms: npt.NDArray[np.float32], waveforms_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        """Convert waveforms to model features."""
        return self._preprocessor.run(["features", "features_lens"], {"waveforms": waveforms, "waveforms_lens": waveforms_lens})
