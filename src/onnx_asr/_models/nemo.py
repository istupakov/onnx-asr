from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

from onnx_asr.asr import Asr, _CtcAsr, _RnntAsr


class NemoConformer(Asr):
    def __init__(self, model_parts: dict[str, Path]):
        super().__init__("nemo", model_parts["vocab"])

    @staticmethod
    def _get_model_parts() -> dict[str, str]:
        return {"vocab": "vocab.txt"}


class NemoConformerCtc(_CtcAsr, NemoConformer):
    def __init__(self, model_parts: dict[str, Path]):
        super().__init__(model_parts)
        self._model = rt.InferenceSession(model_parts["model"])

    @staticmethod
    def _get_model_parts() -> dict[str, str]:
        return {"model": "{model_name}.onnx"} | NemoConformer._get_model_parts()

    def _encode(
        self, features: npt.NDArray[np.float32], features_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        (log_probs,) = self._model.run(["logprobs"], {"audio_signal": features, "length": features_lens})
        return log_probs, (features_lens - 1) // 8 + 1


class NemoConformerRnnt(_RnntAsr, NemoConformer):
    PRED_HIDDEN = 640
    MAX_TOKENS_PER_STEP = 10
    STATE_TYPE = tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]

    def __init__(self, model_parts: dict[str, Path]):
        super().__init__(model_parts)
        self._encoder = rt.InferenceSession(model_parts["encoder"])
        self._decoder_joint = rt.InferenceSession(model_parts["decoder_joint"])

    @staticmethod
    def _get_model_parts() -> dict[str, str]:
        return {
            "encoder": "encoder-{model_name}.onnx",
            "decoder_joint": "decoder_joint-{model_name}.onnx",
        } | NemoConformer._get_model_parts()

    @property
    def _max_tokens_per_step(self) -> int:
        return self.MAX_TOKENS_PER_STEP

    def _encode(
        self, features: npt.NDArray[np.float32], features_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        encoder_out, encoder_out_lens = self._encoder.run(
            ["outputs", "encoded_lengths"], {"audio_signal": features, "length": features_lens}
        )
        return encoder_out, encoder_out_lens

    def _create_state(self) -> STATE_TYPE:
        return (
            np.zeros(shape=(1, 1, self.PRED_HIDDEN), dtype=np.float32),
            np.zeros(shape=(1, 1, self.PRED_HIDDEN), dtype=np.float32),
        )

    def _decode(
        self, prev_tokens: list[int], prev_state: STATE_TYPE, encoder_out: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], STATE_TYPE]:
        outputs, *state = self._decoder_joint.run(
            ["outputs", "output_states_1", "output_states_2"],
            {
                "encoder_outputs": encoder_out[None, :, None],
                "targets": [[[self._blank_idx, *prev_tokens][-1]]],
                "target_length": [1],
                "input_states_1": prev_state[0],
                "input_states_2": prev_state[1],
            },
        )
        return np.squeeze(outputs), tuple(state)
