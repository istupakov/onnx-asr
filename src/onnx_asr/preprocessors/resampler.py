"""Waveform resampler implementations."""

from importlib.resources import files
from typing import Literal

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

from onnx_asr.utils import OnnxSessionOptions, SampleRates, is_float32_array, is_int64_array


class Resampler:
    """Waveform resampler to 8/16 kHz implementation."""

    def __init__(self, sample_rate: Literal[8_000, 16_000], onnx_options: OnnxSessionOptions):
        """Create waveform resampler.

        Args:
            sample_rate: Target sample rate.
            onnx_options: Options for onnxruntime InferenceSession.

        """
        if onnx_options.get("cpu_preprocessing", False):
            onnx_options = {"sess_options": onnx_options.get("sess_options")}
        self._target_sample_rate = sample_rate
        self._preprocessor = rt.InferenceSession(
            files(__package__).joinpath(f"resample{sample_rate // 1000}.onnx").read_bytes(), **onnx_options
        )

    def __call__(
        self, waveforms: npt.NDArray[np.float32], waveforms_lens: npt.NDArray[np.int64], sample_rate: SampleRates
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        """Resample waveform."""
        if sample_rate == self._target_sample_rate:
            return waveforms, waveforms_lens

        resampled, resampled_lens = self._preprocessor.run(
            ["resampled", "resampled_lens"],
            {"waveforms": waveforms, "waveforms_lens": waveforms_lens, "sample_rate": np.array([sample_rate], dtype=np.int64)},
        )
        assert is_float32_array(resampled)
        assert is_int64_array(resampled_lens)
        return resampled, resampled_lens
