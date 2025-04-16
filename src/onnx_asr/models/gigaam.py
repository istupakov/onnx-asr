import numpy as np
import numpy.typing as npt
import onnxruntime as rt
from .. import Preprocessor
from ..asr import Asr, CtcAsr, RnntAsr


class GigaamV2(Asr):
    def __init__(self):
        self._preprocessor = Preprocessor("gigaam")
        self._vocab = dict(enumerate([" "] + [chr(ord("а") + i) for i in range(32)]))

    @property
    def _blank_token_idx(self) -> int:
        return 33

    @property
    def _vocabulary(self) -> dict[int, str]:
        return self._vocab


class GigaamV2Ctc(CtcAsr, GigaamV2):
    def __init__(self, model_path: str):
        super().__init__()
        self._model = rt.InferenceSession(model_path)

    def _encode(
        self, waveforms: npt.NDArray[np.float32], waveforms_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        features, features_lens = self._preprocessor(waveforms, waveforms_lens)
        (log_probs,) = self._model.run(["log_probs"], {"features": features, "feature_lengths": features_lens})
        return log_probs, (features_lens - 1) // 4 + 1


class GigaamV2Rnnt(RnntAsr, GigaamV2):
    PRED_HIDDEN = 320
    STATE_TYPE = tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]

    def __init__(self, encoder_path: str, decoder_path: str, joiner_path: str):
        super().__init__()
        self._encoder = rt.InferenceSession(encoder_path)
        self._decoder = rt.InferenceSession(decoder_path)
        self._joiner = rt.InferenceSession(joiner_path)

    @property
    def _max_tokens_per_step(self) -> int:
        return 3

    def _encode(
        self, waveforms: npt.NDArray[np.float32], waveforms_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        features, features_lens = self._preprocessor(waveforms, waveforms_lens)
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
            ["dec", "h", "c"], {"x": [[[self._blank_token_idx, *prev_tokens][-1]]], "h.1": prev_state[0], "c.1": prev_state[1]}
        )
        (joint,) = self._joiner.run(["joint"], {"enc": encoder_out[None, :, None], "dec": decoder_out.transpose(0, 2, 1)})
        return np.squeeze(joint), tuple(state)
