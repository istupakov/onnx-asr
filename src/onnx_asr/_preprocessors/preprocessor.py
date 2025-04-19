from importlib.resources import files
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import onnxruntime as rt


class Preprocessor:
    PreprocessorNames = Literal["gigaam", "kaldi", "nemo"]

    def __init__(self, name: PreprocessorNames):
        self._preprocessor = rt.InferenceSession(files(__package__).joinpath(Path(name).with_suffix(".onnx")))  # type: ignore

    def __call__(
        self, waveforms: npt.NDArray[np.float32], waveforms_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        return self._preprocessor.run(["features", "features_lens"], {"waveforms": waveforms, "waveforms_lens": waveforms_lens})
