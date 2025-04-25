"""NeMo model implementations."""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

from onnx_asr.asr import _AsrWithCtcDecoding, _AsrWithDecoding, _AsrWithRnntDecoding


class _NemoConformer(_AsrWithDecoding):
    def __init__(self, model_files: dict[str, Path], **kwargs: Any):
        super().__init__("nemo", model_files["vocab"], **kwargs)

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        return {"vocab": "vocab*.txt"}


class NemoConformerCtc(_AsrWithCtcDecoding, _NemoConformer):
    """NeMo Conformer CTC model implementations."""

    def __init__(self, model_files: dict[str, Path], **kwargs: Any):
        """Create NeMo Conformer CTC model.

        Args:
            model_files: Dict with paths to model files.
            kwargs: Additional parameters for onnxruntime.InferenceSession.

        """
        super().__init__(model_files, **kwargs)
        self._model = rt.InferenceSession(model_files["model"], **kwargs)

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else "[!0-9]"
        return {"model": f"stt_*conformer*{suffix}.onnx"} | _NemoConformer._get_model_files(quantization)

    def _encode(
        self, features: npt.NDArray[np.float32], features_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        (log_probs,) = self._model.run(["logprobs"], {"audio_signal": features, "length": features_lens})
        conformer_lens = (features_lens - 1) // 4 + 1
        fastconformer_lens = (features_lens - 1) // 8 + 1
        assert log_probs.shape[1] == max(conformer_lens) or log_probs.shape[1] == max(fastconformer_lens)
        if log_probs.shape[1] == max(conformer_lens):
            return log_probs, conformer_lens
        else:
            return log_probs, fastconformer_lens


class NemoConformerRnnt(_AsrWithRnntDecoding, _NemoConformer):
    """NeMo Conformer RNN-T model implementations."""

    PRED_HIDDEN = 640
    MAX_TOKENS_PER_STEP = 10
    STATE_TYPE = tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]

    def __init__(self, model_files: dict[str, Path], **kwargs: Any):
        """Create NeMo Conformer RNN-T model.

        Args:
            model_files: Dict with paths to model files.
            kwargs: Additional parameters for onnxruntime.InferenceSession.

        """
        super().__init__(model_files, **kwargs)
        self._encoder = rt.InferenceSession(model_files["encoder"], **kwargs)
        self._decoder_joint = rt.InferenceSession(model_files["decoder_joint"], **kwargs)

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else "[!0-9]"
        return {
            "encoder": f"encoder-stt_*conformer*{suffix}.onnx",
            "decoder_joint": f"decoder_joint-stt_*conformer*{suffix}.onnx",
        } | _NemoConformer._get_model_files(quantization)

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
