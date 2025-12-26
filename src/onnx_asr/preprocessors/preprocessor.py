"""ASR preprocessor implementations."""

from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

from onnx_asr.onnx import OnnxSessionOptions, TensorRtOptions, get_onnx_providers
from onnx_asr.utils import is_float32_array, is_int64_array


class PreprocessorRuntimeConfig(OnnxSessionOptions, total=False):
    """Preprocessor runtime config."""

    max_concurrent_workers: int | None
    """Max parallel preprocessing threads (None - auto, 1 - without parallel processing)."""


class Preprocessor:
    """ASR preprocessor implementation."""

    def __init__(self, name: str, runtime_config: PreprocessorRuntimeConfig):
        """Create ASR preprocessor.

        Args:
            name: Preprocessor name.
            runtime_config: Runtime configuration.

        """
        onnx_options = runtime_config.copy()
        self._max_concurrent_workers = onnx_options.pop("max_concurrent_workers", 1)
        if name == "identity":
            self._preprocessor = None
        else:
            providers = get_onnx_providers(onnx_options)
            if name == "kaldi" and providers and providers != ["CPUExecutionProvider"]:
                name = "kaldi_fast"

            filename = str(Path(name).with_suffix(".onnx"))
            self._preprocessor = rt.InferenceSession(
                files(__package__).joinpath(filename).read_bytes(),
                **TensorRtOptions.add_profile(onnx_options, self._preprocessor_shapes),
            )

    @staticmethod
    def _get_excluded_providers() -> list[str]:
        return ["CUDAExecutionProvider"]

    def _preprocessor_shapes(self, waveform_len_ms: int, **kwargs: int) -> str:
        return "waveforms:{batch}x{len},waveforms_lens:{batch}".format(len=waveform_len_ms * 16, **kwargs)

    def _preprocess(
        self, waveforms: npt.NDArray[np.float32], waveforms_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        if not self._preprocessor:
            return waveforms, waveforms_lens

        features, features_lens = self._preprocessor.run(
            ["features", "features_lens"], {"waveforms": waveforms, "waveforms_lens": waveforms_lens}
        )
        assert is_float32_array(features)
        assert is_int64_array(features_lens)
        return features, features_lens

    def __call__(
        self, waveforms: npt.NDArray[np.float32], waveforms_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        """Convert waveforms to model features."""
        if self._preprocessor is None or waveforms.shape[0] == 1 or self._max_concurrent_workers == 1:
            return self._preprocess(waveforms, waveforms_lens)

        with ThreadPoolExecutor(max_workers=self._max_concurrent_workers) as executor:
            features, features_lens = zip(
                *executor.map(self._preprocess, waveforms[:, None], waveforms_lens[:, None]), strict=True
            )
        return np.concatenate(features, axis=0), np.concatenate(features_lens, axis=0)
