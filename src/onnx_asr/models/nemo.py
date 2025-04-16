import numpy as np
import numpy.typing as npt
import onnxruntime as rt
from .. import Preprocessor
from ..asr import Asr, CtcAsr, RnntAsr


class NemoConformer(Asr):
    def __init__(self, vocab_path: str):
        self._preprocessor = Preprocessor("nemo")
        self._vocab = dict(np.genfromtxt(vocab_path, dtype=None, delimiter=" ", usecols=[1, 0]).tolist())

    @property
    def _blank_token_idx(self) -> int:
        return len(self._vocab)

    @property
    def _vocabulary(self) -> dict[int, str]:
        return self._vocab


class NemoConformerCtc(CtcAsr, NemoConformer):
    def __init__(self, model_path: str, vocab_path: str):
        super().__init__(vocab_path)
        self._model = rt.InferenceSession(model_path)

    def _encode(
        self, waveforms: npt.NDArray[np.float32], waveforms_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        features, features_lens = self._preprocessor(waveforms, waveforms_lens)
        (log_probs,) = self._model.run(["logprobs"], {"audio_signal": features, "length": features_lens})
        return log_probs, (features_lens - 1) // 8 + 1


class NemoConformerRnnt(RnntAsr, NemoConformer):
    PRED_HIDDEN = 640
    MAX_TOKENS_PER_STEP = 10
    STATE_TYPE = tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]

    def __init__(self, encoder_path: str, decoder_joint_path: str, vocab_path: str):
        super().__init__(vocab_path)
        self._encoder = rt.InferenceSession(encoder_path)
        self._decoder_joint = rt.InferenceSession(decoder_joint_path)

    @property
    def _max_tokens_per_step(self) -> int:
        return self.MAX_TOKENS_PER_STEP

    def _encode(
        self, waveforms: npt.NDArray[np.float32], waveforms_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        features, features_lens = self._preprocessor(waveforms, waveforms_lens)
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
                "targets": [[[self._blank_token_idx, *prev_tokens][-1]]],
                "target_length": [1],
                "input_states_1": prev_state[0],
                "input_states_2": prev_state[1],
            },
        )
        return np.squeeze(outputs), tuple(state)
