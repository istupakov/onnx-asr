"""Waveform resampler implementations."""

from importlib.resources import files
from typing import get_args

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

from onnx_asr.utils import OnnxSessionOptions, SampleRates, is_float32_array, is_int64_array


class Resampler:
    """Waveform resampler to 16 kHz implementation."""

    def __init__(self, onnx_options: OnnxSessionOptions):
        """Create waveform resampler.

        Args:
            onnx_options: Options for onnxruntime InferenceSession.

        """
        if onnx_options.get("cpu_preprocessing", False):
            onnx_options = {"sess_options": onnx_options.get("sess_options")}
        self._preprocessors = {}
        for sample_rate in get_args(SampleRates):
            if sample_rate == 16_000:
                continue
            self._preprocessors[sample_rate] = rt.InferenceSession(
                files(__package__).joinpath(f"resample{sample_rate // 1000}.onnx").read_bytes(), **onnx_options
            )

    def __call__(
        self, waveforms: npt.NDArray[np.float32], waveforms_lens: npt.NDArray[np.int64], sample_rate: SampleRates
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        """Resample waveform to 16 kHz."""
        if sample_rate == 16_000:
            return waveforms, waveforms_lens

        resampled, resampled_lens = self._preprocessors[sample_rate].run(
            ["resampled", "resampled_lens"],
            {"waveforms": waveforms, "waveforms_lens": waveforms_lens},
        )
        assert is_float32_array(resampled) and is_int64_array(resampled_lens)
        return resampled, resampled_lens
