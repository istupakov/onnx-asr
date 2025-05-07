"""Waveform resampler implementations."""

from importlib.resources import files

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

from onnx_asr.utils import OnnxSessionOptions, SampleRates


class Resampler:
    """Waveform resampler to 16 kHz implementation."""

    def __init__(self, onnx_options: OnnxSessionOptions):
        """Create waveform resampler.

        Args:
            onnx_options: Options for onnxruntime InferenceSession.

        """
        self._preprocessor = rt.InferenceSession(files(__package__).joinpath("resample.onnx").read_bytes(), **onnx_options)

    def __call__(
        self, waveforms: npt.NDArray[np.float32], waveforms_lens: npt.NDArray[np.int64], sample_rate: SampleRates
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        """Resample waveform to 16 kHz."""
        if sample_rate != 16_000:
            waveforms, waveforms_lens = self._preprocessor.run(
                ["resampled", "resampled_lens"],
                {"waveforms": waveforms, "waveforms_lens": waveforms_lens, "sample_rate": [sample_rate]},
            )
        return waveforms, waveforms_lens
