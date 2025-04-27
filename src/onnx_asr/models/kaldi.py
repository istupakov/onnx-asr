"""Kaldi model implementations."""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

from onnx_asr.asr import _AsrWithRnntDecoding


class KaldiTransducer(_AsrWithRnntDecoding):
    """Kaldi Transducer model implementation."""

    CONTEXT_SIZE = 2

    def __init__(self, model_files: dict[str, Path], **kwargs: Any):
        """Create Kaldi Transducer model.

        Args:
            model_files: Dict with paths to model files.
            kwargs: Additional parameters for onnxruntime.InferenceSession.

        """
        super().__init__("kaldi", model_files["vocab"], **kwargs)
        self._encoder = rt.InferenceSession(model_files["encoder"], **kwargs)
        self._decoder = rt.InferenceSession(model_files["decoder"], **kwargs)
        self._joiner = rt.InferenceSession(model_files["joiner"], **kwargs)

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
    def _max_tokens_per_step(self) -> int:
        return 1

    def _encode(
        self, features: npt.NDArray[np.float32], features_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        encoder_out, encoder_out_lens = self._encoder.run(
            ["encoder_out", "encoder_out_lens"], {"x": features, "x_lens": features_lens}
        )
        return encoder_out.transpose(0, 2, 1), encoder_out_lens

    def _create_state(self) -> tuple:
        return ()

    def _decode(
        self, prev_tokens: list[int], prev_state: tuple, encoder_out: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], tuple]:
        (decoder_out,) = self._decoder.run(["decoder_out"], {"y": [[-1, self._blank_idx, *prev_tokens][-self.CONTEXT_SIZE :]]})
        (logit,) = self._joiner.run(["logit"], {"encoder_out": encoder_out[None, :], "decoder_out": decoder_out})
        return np.squeeze(logit), prev_state


class KaldiTransducerWithCache(KaldiTransducer):
    """Kaldi Transducer (with decoder cache) model implementation."""

    STATE_TYPE = tuple[dict[tuple[int, ...], npt.NDArray[np.float32]]]

    def _create_state(self) -> STATE_TYPE:
        return ({},)

    def _decode(
        self, prev_tokens: list[int], prev_state: STATE_TYPE, encoder_out: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], STATE_TYPE]:
        context = tuple([-1, self._blank_idx, *prev_tokens][-self.CONTEXT_SIZE :])

        decoder_out = prev_state[0].get(context)
        if decoder_out is None:
            (decoder_out,) = self._decoder.run(["decoder_out"], {"y": [context]})
            prev_state[0][context] = decoder_out

        (logit,) = self._joiner.run(["logit"], {"encoder_out": encoder_out[None, :], "decoder_out": decoder_out})
        return np.squeeze(logit), prev_state
