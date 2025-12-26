"""Waveform resampler implementations."""

from importlib.resources import files
from typing import Literal, get_args

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

from onnx_asr.onnx import OnnxSessionOptions, TensorRtOptions
from onnx_asr.utils import SampleRates, is_float32_array, is_int64_array


class Resampler:
    """Waveform resampler to 8/16 kHz implementation."""

    def __init__(self, sample_rate: Literal[8_000, 16_000], onnx_options: OnnxSessionOptions):
        """Create waveform resampler.

        Args:
            sample_rate: Target sample rate.
            onnx_options: Options for onnxruntime InferenceSession.

        """
        self._target_sample_rate = sample_rate
        self._preprocessors = {}
        for orig_freq in get_args(SampleRates):
            if orig_freq == sample_rate:
                continue
            self._preprocessors[orig_freq] = rt.InferenceSession(
                files(__package__).joinpath(f"resample_{orig_freq // 1000}_{sample_rate // 1000}.onnx").read_bytes(),
                **TensorRtOptions.add_profile(onnx_options, self._preprocessor_shapes),
            )

    @staticmethod
    def _get_excluded_providers() -> list[str]:
        return TensorRtOptions.get_provider_names()

    def _preprocessor_shapes(self, waveform_len_ms: int, **kwargs: int) -> str:
        return "waveforms:{batch}x{len},waveforms_lens:{batch}".format(
            len=kwargs.get("resampler_waveform_len_ms", waveform_len_ms) * 48, **kwargs
        )

    def __call__(
        self, waveforms: npt.NDArray[np.float32], waveforms_lens: npt.NDArray[np.int64], sample_rate: SampleRates
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        """Resample waveform."""
        if sample_rate == self._target_sample_rate:
            return waveforms, waveforms_lens

        resampled, resampled_lens = self._preprocessors[sample_rate].run(
            ["resampled", "resampled_lens"],
            {"waveforms": waveforms, "waveforms_lens": waveforms_lens},
        )
        assert is_float32_array(resampled)
        assert is_int64_array(resampled_lens)
        return resampled, resampled_lens
