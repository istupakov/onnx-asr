"""T-one model implementations."""

from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

from onnx_asr.asr import AsrRuntimeConfig, _AsrWithCtcDecoding
from onnx_asr.onnx import TensorRtOptions
from onnx_asr.utils import is_float16_array, is_float32_array


class TOneCtc(_AsrWithCtcDecoding):
    """T-one CTC model implementation."""

    def __init__(self, model_files: dict[str, Path], runtime_config: AsrRuntimeConfig):
        """Create T-one CTC model.

        Args:
            model_files: Dict with paths to model files.
            runtime_config: Runtime configuration.

        """
        super().__init__(model_files, runtime_config)
        self._model = rt.InferenceSession(
            model_files["model"], **TensorRtOptions.add_profile(runtime_config.onnx_options, self._encoder_shapes)
        )

        shapes = {x.name: x.shape for x in self._model.get_inputs()}
        self._chunk_size = shapes["signal"][1]
        self._state_size = shapes["state"][1]

        self._vocab: dict[int, str] = dict(enumerate(self.config["decoder_params"]["vocabulary"]))  # type: ignore[typeddict-item]
        self._vocab_size = len(self._vocab) + 1
        self._blank_idx = int(self.config["pad_token_id"])  # type: ignore[typeddict-item]

    @staticmethod
    def _get_excluded_providers() -> list[str]:
        return ["CoreMLExecutionProvider"]

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        return {"model": f"model{suffix}.onnx"}

    @staticmethod
    def _get_sample_rate() -> Literal[8_000, 16_000]:
        return 8_000

    @property
    def _preprocessor_name(self) -> str:
        return "identity"

    @property
    def _subsampling_factor(self) -> int:
        return int(self.config["encoder_params"]["reduction_kernel_size"])  # type: ignore[typeddict-item]

    def _encoder_shapes(self, **kwargs: int) -> str:
        return "signal:{batch}x2400x1,state:{batch}x219729".format(**kwargs)

    def _encode_chunk(
        self, waveforms: npt.NDArray[np.float32], state: npt.NDArray[np.float16]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float16]]:
        (logprobs, new_state) = self._model.run(
            ["logprobs", "state_next"], {"signal": (waveforms[..., None] * (2**15 - 1)).astype(np.int32), "state": state}
        )
        assert is_float32_array(logprobs)
        assert is_float16_array(new_state)
        return logprobs, new_state

    def _encode(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        waveforms = np.pad(waveforms, ((0, 0), (self._chunk_size, self._chunk_size + (-waveforms.shape[1]) % self._chunk_size)))

        res = []
        state = np.zeros((waveforms.shape[0], self._state_size), dtype=np.float16)
        for chunk in np.split(waveforms, waveforms.shape[1] // self._chunk_size, axis=1):
            logprobs, state = self._encode_chunk(chunk, state)
            res.append(logprobs)

        return np.hstack(res[1:]), res[0].shape[1] * ((waveforms_len + self._chunk_size - 1) // self._chunk_size + 1)
