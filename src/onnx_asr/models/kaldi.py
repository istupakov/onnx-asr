import numpy as np
import numpy.typing as npt
import onnxruntime as rt
from .. import Preprocessor
from ..asr import RnntAsr


class KaldiTransducer(RnntAsr):
    CONTEXT_SIZE = 2

    def __init__(self, encoder_path: str, decoder_path: str, joiner_path: str, vocab_path: str):
        self._preprocessor = Preprocessor("kaldi")
        self._vocab = dict(np.genfromtxt(vocab_path, dtype=None, delimiter=" ", usecols=[1, 0]).tolist())
        self._encoder = rt.InferenceSession(encoder_path)
        self._decoder = rt.InferenceSession(decoder_path)
        self._joiner = rt.InferenceSession(joiner_path)

    @property
    def _blank_token_idx(self) -> int:
        return 0

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
