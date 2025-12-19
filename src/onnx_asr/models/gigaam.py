"""GigaAM v2+ model implementations."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

from onnx_asr.asr import AsrRuntimeConfig, _AsrWithCtcDecoding, _AsrWithDecoding, _AsrWithTransducerDecoding
from onnx_asr.onnx import TensorRtOptions
from onnx_asr.utils import is_float32_array, is_int32_array


class _GigaamV2(_AsrWithDecoding):
    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        return {"vocab": "v?_vocab.txt"}

    @property
    def _preprocessor_name(self) -> str:
        assert self.config.get("features_size", 64) == 64
        version = self.config.get("version", "v2")
        return f"gigaam_{version}"

    @property
    def _subsampling_factor(self) -> int:
        return self.config.get("subsampling_factor", 4)


class GigaamV2Ctc(_AsrWithCtcDecoding, _GigaamV2):
    """GigaAM v2+ CTC model implementation."""

    def __init__(self, model_files: dict[str, Path], runtime_config: AsrRuntimeConfig):
        """Create GigaAM v2+ CTC model.

        Args:
            model_files: Dict with paths to model files.
            runtime_config: Runtime configuration.

        """
        super().__init__(model_files, runtime_config)
        self._model = rt.InferenceSession(
            model_files["model"], **TensorRtOptions.add_profile(runtime_config.onnx_options, self._encoder_shapes)
        )

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        return {"model": f"v?_ctc{suffix}.onnx"} | _GigaamV2._get_model_files(quantization)

    def _encoder_shapes(self, waveform_len_ms: int, **kwargs: int) -> str:
        return "features:{batch}x64x{len},feature_lengths:{batch}".format(len=waveform_len_ms // 10, **kwargs)

    def _encode(
        self, features: npt.NDArray[np.float32], features_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        (log_probs,) = self._model.run(["log_probs"], {"features": features, "feature_lengths": features_lens})
        assert is_float32_array(log_probs)
        return log_probs, (features_lens - 1) // self._subsampling_factor + 1


_STATE_TYPE = list[npt.NDArray[np.float32]]


class GigaamV2Rnnt(_AsrWithTransducerDecoding[_STATE_TYPE], _GigaamV2):
    """GigaAM v2+ RNN-T model implementation."""

    PRED_HIDDEN = 320

    def __init__(self, model_files: dict[str, Path], runtime_config: AsrRuntimeConfig):
        """Create GigaAM v2+ RNN-T model.

        Args:
            model_files: Dict with paths to model files.
            runtime_config: Runtime configuration.

        """
        super().__init__(model_files, runtime_config)
        self._encoder = rt.InferenceSession(
            model_files["encoder"], **TensorRtOptions.add_profile(runtime_config.onnx_options, self._encoder_shapes)
        )
        self._decoder = rt.InferenceSession(model_files["decoder"], **runtime_config.onnx_options)
        self._joiner = rt.InferenceSession(model_files["joint"], **runtime_config.onnx_options)

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        return {
            "encoder": f"v?_rnnt_encoder{suffix}.onnx",
            "decoder": f"v?_rnnt_decoder{suffix}.onnx",
            "joint": f"v?_rnnt_joint{suffix}.onnx",
        } | _GigaamV2._get_model_files(quantization)

    @property
    def _max_tokens_per_step(self) -> int:
        return self.config.get("max_tokens_per_step", 3)

    def _encoder_shapes(self, waveform_len_ms: int, **kwargs: int) -> str:
        return "audio_signal:{batch}x64x{len},length:{batch}".format(len=waveform_len_ms // 10, **kwargs)

    def _encode(
        self, features: npt.NDArray[np.float32], features_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        encoder_out, encoder_out_lens = self._encoder.run(
            ["encoded", "encoded_len"], {"audio_signal": features, "length": features_lens}
        )
        assert is_float32_array(encoder_out)
        assert is_int32_array(encoder_out_lens)
        return encoder_out.transpose(0, 2, 1), encoder_out_lens.astype(np.int64)

    def _create_state(self) -> _STATE_TYPE:
        return [
            np.zeros(shape=(1, 1, self.PRED_HIDDEN), dtype=np.float32),
            np.zeros(shape=(1, 1, self.PRED_HIDDEN), dtype=np.float32),
        ]

    def _decode(
        self, prev_tokens: list[int], prev_state: _STATE_TYPE, encoder_out: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], int, _STATE_TYPE]:
        if len(prev_state) == 2:
            decoder_out, state1, state2 = self._decoder.run(
                ["dec", "h", "c"],
                {"x": [[prev_tokens[-1] if prev_tokens else self._blank_idx]], "h.1": prev_state[0], "c.1": prev_state[1]},
            )
            assert is_float32_array(decoder_out)
            assert is_float32_array(state1)
            assert is_float32_array(state2)
            prev_state[:] = (decoder_out, state1, state2)
        else:
            decoder_out, state1, state2 = prev_state

        (joint,) = self._joiner.run(["joint"], {"enc": encoder_out[None, :, None], "dec": decoder_out.transpose(0, 2, 1)})
        assert is_float32_array(joint)
        return np.squeeze(joint), -1, [state1, state2]


class GigaamV3E2eCtc(GigaamV2Ctc):
    """GigaAM v3 E2E CTC model implementation."""

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        return {"model": f"v3_e2e_ctc{suffix}.onnx", "vocab": "v3_e2e_ctc_vocab.txt"}


class GigaamV3E2eRnnt(GigaamV2Rnnt):
    """GigaAM v3 E2E RNN-T model implementation."""

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        return {
            "encoder": f"v3_e2e_rnnt_encoder{suffix}.onnx",
            "decoder": f"v3_e2e_rnnt_decoder{suffix}.onnx",
            "joint": f"v3_e2e_rnnt_joint{suffix}.onnx",
            "vocab": "v3_e2e_rnnt_vocab.txt",
        }
