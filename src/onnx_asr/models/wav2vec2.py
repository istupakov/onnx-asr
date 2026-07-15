"""Wav2Vec2 CTC model implementation."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

from onnx_asr.asr import Preprocessor, _AsrWithCtcDecoding
from onnx_asr.onnx import OnnxSessionOptions
from onnx_asr.utils import is_float32_array, is_int64_array


class Wav2Vec2Ctc(_AsrWithCtcDecoding):
    """Wav2Vec2 CTC model implementation (HuggingFace wav2vec2 / XLS-R fine-tunes)."""

    def __init__(  # noqa: D107
        self,
        model_files: dict[str, Path],
        preprocessor_factory: Callable[[str], Preprocessor],
        onnx_options: OnnxSessionOptions,
    ):
        super().__init__(model_files, preprocessor_factory, onnx_options)
        self._model = rt.InferenceSession(model_files["model"], **onnx_options)

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        return {"model": f"model{suffix}.onnx", "vocab": "vocab.txt"}

    @property
    def _preprocessor_name(self) -> str:
        return "identity"

    @property
    def _subsampling_factor(self) -> int:
        return int(self.config.get("subsampling_factor", 320))

    def _encode(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        (logprobs,) = self._model.run(
            ["logprobs"],
            {"input_values": waveforms, "input_lengths": waveforms_len.astype(np.int64)},
        )
        assert is_float32_array(logprobs)
        out_lens = waveforms_len // self._subsampling_factor + 1
        out_lens = np.minimum(out_lens, logprobs.shape[1]).astype(np.int64)
        assert is_int64_array(out_lens)
        return logprobs, out_lens
