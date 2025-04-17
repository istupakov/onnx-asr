import numpy as np
import numpy.typing as npt
import onnxruntime as rt
from pathlib import Path

from ..asr import RnntAsr
from ..preprocessors import Preprocessor


class KaldiTransducer(RnntAsr):
    CONTEXT_SIZE = 2

    def __init__(self, model_parts: dict[str, Path]):
        self._preprocessor = Preprocessor("kaldi")
        self._vocab = dict(np.genfromtxt(model_parts["vocab"], dtype=None, delimiter=" ", usecols=[1, 0]).tolist())
        self._blank_idx = next(key for (key, value) in self._vocab.items() if value == "<blk>")
        self._encoder = rt.InferenceSession(model_parts["encoder"])
        self._decoder = rt.InferenceSession(model_parts["decoder"])
        self._joiner = rt.InferenceSession(model_parts["joiner"])

    @staticmethod
    def _get_model_parts() -> dict[str, str]:
        return {"encoder": "encoder.onnx", "decoder": "decoder.onnx", "joiner": "joiner.onnx", "vocab": "tokens.txt"}

    @property
    def _blank_token_idx(self) -> int:
        return self._blank_idx

    @property
    def _max_tokens_per_step(self) -> int:
        return 1

    @property
    def _vocabulary(self) -> dict[int, str]:
        return self._vocab

    def _encode(
        self, waveforms: npt.NDArray[np.float32], waveforms_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        features, features_lens = self._preprocessor(waveforms, waveforms_lens)
        encoder_out, encoder_out_lens = self._encoder.run(
            ["encoder_out", "encoder_out_lens"], {"x": features, "x_lens": features_lens}
        )
        return encoder_out.transpose(0, 2, 1), encoder_out_lens

    def _create_state(self) -> None:
        return None

    def _decode(
        self, prev_tokens: list[int], prev_state: None, encoder_out: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], None]:
        (decoder_out,) = self._decoder.run(
            ["decoder_out"], {"y": [[-1, self._blank_token_idx, *prev_tokens][-self.CONTEXT_SIZE :]]}
        )
        (logit,) = self._joiner.run(["logit"], {"encoder_out": encoder_out[None, :], "decoder_out": decoder_out})
        return np.squeeze(logit), None
