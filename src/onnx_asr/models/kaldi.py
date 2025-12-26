"""Kaldi model implementations."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

from onnx_asr.asr import AsrRuntimeConfig, _AsrWithTransducerDecoding
from onnx_asr.onnx import TensorRtOptions
from onnx_asr.utils import is_float32_array, is_int64_array

_STATE_TYPE = dict[tuple[int, ...], npt.NDArray[np.float32]]


class KaldiTransducer(_AsrWithTransducerDecoding[_STATE_TYPE]):
    """Kaldi Transducer model implementation."""

    CONTEXT_SIZE = 2

    def __init__(self, model_files: dict[str, Path], runtime_config: AsrRuntimeConfig):
        """Create Kaldi Transducer model.

        Args:
            model_files: Dict with paths to model files.
            runtime_config: Runtime configuration.

        """
        super().__init__(model_files, runtime_config)
        self._encoder = rt.InferenceSession(
            model_files["encoder"], **TensorRtOptions.add_profile(runtime_config.onnx_options, self._encoder_shapes)
        )
        self._decoder = rt.InferenceSession(model_files["decoder"], **runtime_config.onnx_options)
        self._joiner = rt.InferenceSession(model_files["joiner"], **runtime_config.onnx_options)

    @staticmethod
    def _get_excluded_providers() -> list[str]:
        return TensorRtOptions.get_provider_names()

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        return {
            "encoder": f"*/encoder{suffix}.onnx",
            "decoder": f"*/decoder{suffix}.onnx",
            "joiner": f"*/joiner{suffix}.onnx",
            "vocab": "*/tokens.txt",
        }

    @property
    def _preprocessor_name(self) -> str:
        assert self.config.get("features_size", 80) == 80
        return "kaldi"

    @property
    def _subsampling_factor(self) -> int:
        return self.config.get("subsampling_factor", 4)

    @property
    def _max_tokens_per_step(self) -> int:
        return self.config.get("max_tokens_per_step", 1)

    def _encoder_shapes(self, waveform_len_ms: int, **kwargs: int) -> str:
        return "x:{batch}x{len}x80,x_lens:{batch}".format(len=waveform_len_ms // 10, **kwargs)

    def _encode(
        self, features: npt.NDArray[np.float32], features_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        encoder_out, encoder_out_lens = self._encoder.run(
            ["encoder_out", "encoder_out_lens"], {"x": features, "x_lens": features_lens}
        )
        assert is_float32_array(encoder_out)
        assert is_int64_array(encoder_out_lens)
        return encoder_out, encoder_out_lens

    def _create_state(self) -> _STATE_TYPE:
        return {}

    def _decode(
        self, prev_tokens: list[int], prev_state: _STATE_TYPE, encoder_out: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], int, _STATE_TYPE]:
        context = (-1, self._blank_idx, *prev_tokens)[-self.CONTEXT_SIZE :]

        decoder_out = prev_state.get(context)
        if decoder_out is None:
            (_decoder_out,) = self._decoder.run(["decoder_out"], {"y": [context]})
            assert is_float32_array(_decoder_out)
            prev_state[context] = (decoder_out := _decoder_out)

        (logit,) = self._joiner.run(["logit"], {"encoder_out": encoder_out[None, :], "decoder_out": decoder_out})
        assert is_float32_array(logit)
        return np.squeeze(logit), -1, prev_state
