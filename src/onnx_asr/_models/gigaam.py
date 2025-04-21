from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

from onnx_asr.asr import Asr, _CtcAsr, _RnntAsr


class GigaamV2(Asr):
    def __init__(self, model_parts: dict[str, Path]):
        super().__init__("gigaam", model_parts["vocab"])

    @staticmethod
    def _get_model_files(version: str | None = None) -> dict[str, str]:
        assert version is None, "For now, only the default version is supported."
        return {"vocab": "v2_vocab.txt"}


class GigaamV2Ctc(_CtcAsr, GigaamV2):
    def __init__(self, model_parts: dict[str, Path]):
        super().__init__(model_parts)
        self._model = rt.InferenceSession(model_parts["model"])

    @staticmethod
    def _get_model_files(version: str | None = None) -> dict[str, str]:
        return {"model": "v2_ctc.onnx"} | GigaamV2._get_model_files(version)

    def _encode(
        self, features: npt.NDArray[np.float32], features_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        (log_probs,) = self._model.run(["log_probs"], {"features": features, "feature_lengths": features_lens})
        return log_probs, (features_lens - 1) // 4 + 1


class GigaamV2Rnnt(_RnntAsr, GigaamV2):
    PRED_HIDDEN = 320
    STATE_TYPE = tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]

    def __init__(self, model_parts: dict[str, Path]):
        super().__init__(model_parts)
        self._encoder = rt.InferenceSession(model_parts["encoder"])
        self._decoder = rt.InferenceSession(model_parts["decoder"])
        self._joiner = rt.InferenceSession(model_parts["joint"])

    @staticmethod
    def _get_model_files(version: str | None = None) -> dict[str, str]:
        return {
            "encoder": "v2_rnnt_encoder.onnx",
            "decoder": "v2_rnnt_decoder.onnx",
            "joint": "v2_rnnt_joint.onnx",
        } | GigaamV2._get_model_files(version)

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

    def _create_state(self) -> STATE_TYPE:
        return (
            np.zeros(shape=(1, 1, self.PRED_HIDDEN), dtype=np.float32),
            np.zeros(shape=(1, 1, self.PRED_HIDDEN), dtype=np.float32),
        )

    def _decode(
        self, prev_tokens: list[int], prev_state: STATE_TYPE, encoder_out: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], STATE_TYPE]:
        decoder_out, *state = self._decoder.run(
            ["dec", "h", "c"], {"x": [[[self._blank_idx, *prev_tokens][-1]]], "h.1": prev_state[0], "c.1": prev_state[1]}
        )
        (joint,) = self._joiner.run(["joint"], {"enc": encoder_out[None, :, None], "dec": decoder_out.transpose(0, 2, 1)})
        return np.squeeze(joint), tuple(state)
