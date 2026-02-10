"""ASR preprocessor implementations."""

from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

import onnx_asr.preprocessors
from onnx_asr.asr import Preprocessor
from onnx_asr.onnx import OnnxSessionOptions, TensorRtOptions
from onnx_asr.utils import is_float32_array, is_int64_array


class IdentityPreprocessor:
    """Identity preprocessor implementation."""

    def __call__(
        self, waveforms: npt.NDArray[np.float32], waveforms_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        """Convert waveforms to model features."""
        return waveforms, waveforms_lens


class OnnxPreprocessor:
    """ONNX preprocessor implementation."""

    def __init__(self, name: str, onnx_options: OnnxSessionOptions):
        """Create preprocessor.

        Args:
            name: Preprocessor name.
            onnx_options: Options for onnxruntime InferenceSession.

        """
        filename = str(Path(name).with_suffix(".onnx"))
        self._preprocessor = rt.InferenceSession(
            files(onnx_asr.preprocessors).joinpath("data").joinpath(filename).read_bytes(),
            **TensorRtOptions.add_profile(onnx_options, self._preprocessor_shapes),
        )

    @staticmethod
    def _get_excluded_providers() -> list[str]:
        return ["CUDAExecutionProvider"]

    def _preprocessor_shapes(self, waveform_len_ms: int, **kwargs: int) -> str:
        return "waveforms:{batch}x{len},waveforms_lens:{batch}".format(len=waveform_len_ms * 16, **kwargs)

    def __call__(
        self, waveforms: npt.NDArray[np.float32], waveforms_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        """Convert waveforms to model features."""
        features, features_lens = self._preprocessor.run(
            ["features", "features_lens"], {"waveforms": waveforms, "waveforms_lens": waveforms_lens}
        )
        assert is_float32_array(features)
        assert is_int64_array(features_lens)
        return features, features_lens


class ConcurrentPreprocessor:
    """Concurrent preprocessor implementation."""

    def __init__(self, preprocessor: Preprocessor, max_concurrent_workers: int | None = None):
        """Create preprocessor.

        Args:
            preprocessor: sequential preprocessor.
            max_concurrent_workers: Max concurrent workers for batch processing.

        """
        self.preprocessor = preprocessor
        self._max_concurrent_workers = max_concurrent_workers

    def __call__(
        self, waveforms: npt.NDArray[np.float32], waveforms_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        """Convert waveforms to model features."""
        if waveforms.shape[0] == 1 or self._max_concurrent_workers == 1:
            return self.preprocessor(waveforms, waveforms_lens)

        with ThreadPoolExecutor(max_workers=self._max_concurrent_workers) as executor:
            features, features_lens = zip(
                *executor.map(self.preprocessor, waveforms[:, None], waveforms_lens[:, None]), strict=True
            )
        return np.concatenate(features, axis=0), np.concatenate(features_lens, axis=0)
