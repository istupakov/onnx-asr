"""GigaAM v2 model implementations."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

from onnx_asr.asr import _AsrWithCtcDecoding, _AsrWithDecoding, _AsrWithTransducerDecoding
from onnx_asr.utils import OnnxSessionOptions


class _GigaamV2(_AsrWithDecoding):
    def __init__(self, model_files: dict[str, Path], onnx_options: OnnxSessionOptions):
        super().__init__("gigaam", model_files["vocab"], onnx_options)

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        return {"vocab": "v2_vocab.txt"}


class GigaamV2Ctc(_AsrWithCtcDecoding, _GigaamV2):
    """GigaAM v2 CTC model implementation."""

    def __init__(self, model_files: dict[str, Path], onnx_options: OnnxSessionOptions):
        """Create GigaAM v2 CTC model.

        Args:
            model_files: Dict with paths to model files.
            onnx_options: Options for onnxruntime InferenceSession.

        """
        super().__init__(model_files, onnx_options)
        self._model = rt.InferenceSession(model_files["model"], **onnx_options)

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        return {"model": f"v2_ctc{suffix}.onnx"} | _GigaamV2._get_model_files(quantization)

    def _encode(
        self, features: npt.NDArray[np.float32], features_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        (log_probs,) = self._model.run(["log_probs"], {"features": features, "feature_lengths": features_lens})
        return log_probs, (features_lens - 1) // 4 + 1


_STATE_TYPE = tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]


class GigaamV2Rnnt(_AsrWithTransducerDecoding[_STATE_TYPE], _GigaamV2):
    """GigaAM v2 RNN-T model implementation."""

    PRED_HIDDEN = 320

    def __init__(self, model_files: dict[str, Path], onnx_options: OnnxSessionOptions):
        """Create GigaAM v2 RNN-T model.

        Args:
            model_files: Dict with paths to model files.
            onnx_options: Options for onnxruntime InferenceSession.

        """
        super().__init__(model_files, onnx_options)
        self._encoder = rt.InferenceSession(model_files["encoder"], **onnx_options)
        self._decoder = rt.InferenceSession(model_files["decoder"], **onnx_options)
        self._joiner = rt.InferenceSession(model_files["joint"], **onnx_options)

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        return {
            "encoder": f"v2_rnnt_encoder{suffix}.onnx",
            "decoder": f"v2_rnnt_decoder{suffix}.onnx",
            "joint": f"v2_rnnt_joint{suffix}.onnx",
        } | _GigaamV2._get_model_files(quantization)

    @property
    def _max_tokens_per_step(self) -> int:
        return 3

    def _encode(
        self, features: npt.NDArray[np.float32], features_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        encoder_out, encoder_out_lens = self._encoder.run(
            ["encoded", "encoded_len"], {"audio_signal": features, "length": features_lens}
        )
        return encoder_out, encoder_out_lens.astype(np.int64)

    def _create_state(self) -> _STATE_TYPE:
        return (
            np.zeros(shape=(1, 1, self.PRED_HIDDEN), dtype=np.float32),
            np.zeros(shape=(1, 1, self.PRED_HIDDEN), dtype=np.float32),
        )

    def _decode(
        self, prev_tokens: list[int], prev_state: _STATE_TYPE, encoder_out: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], int, _STATE_TYPE]:
        decoder_out, *state = self._decoder.run(
            ["dec", "h", "c"], {"x": [[[self._blank_idx, *prev_tokens][-1]]], "h.1": prev_state[0], "c.1": prev_state[1]}
        )
        (joint,) = self._joiner.run(["joint"], {"enc": encoder_out[None, :, None], "dec": decoder_out.transpose(0, 2, 1)})
        return np.squeeze(joint), -1, tuple(state)
