import re
import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
from typing import Any
from .utils import pad_list


class Asr(ABC):
    @property
    @abstractmethod
    def _blank_token_idx(self) -> int:
        pass

    @property
    @abstractmethod
    def _vocabulary(self) -> dict[int, str]:
        pass

    @abstractmethod
    def _encode(
        self, waveforms: npt.NDArray[np.float32], waveforms_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        pass

    @abstractmethod
    def recognize_batch(self, waveforms: list[npt.NDArray[np.float32]]) -> list[str]:
        pass

    def _decode_tokens(self, tokens: list[int]) -> str:
        text = "".join([self._vocabulary[i] for i in tokens])
        return re.sub(r"\A\u2581|\u2581\B", "", text).replace("\u2581", " ")

    def recognize(self, waveform: npt.NDArray[np.float32]) -> str:
        return self.recognize_batch([waveform])[0]


class CtcAsr(Asr):
    def recognize_batch(self, waveforms: list[npt.NDArray[np.float32]]) -> list[str]:
        results = []
        for log_probs, len in zip(*self._encode(*pad_list(waveforms))):
            indices = log_probs[:len].argmax(axis=-1)
            indices = indices[np.diff(indices).nonzero()]
            indices = indices[indices != self._blank_token_idx]
            results.append(self._decode_tokens(indices))

        return results


class RnntAsr(Asr):
    @abstractmethod
    def _create_state(self) -> Any:
        pass

    @property
    @abstractmethod
    def _max_tokens_per_step(self) -> int:
        pass

    @abstractmethod
    def _decode(
        self, prev_tokens: list[int], prev_state: Any, encoder_out: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], Any]:
        pass

    def recognize_batch(self, waveforms: list[npt.NDArray[np.float32]]) -> list[str]:
        results = []
        for encodings, len in zip(*self._encode(*pad_list(waveforms))):
            prev_state = self._create_state()
            tokens = []

            for t in range(len):
                emitted_tokens = 0
                while emitted_tokens < self._max_tokens_per_step:
                    probs, state = self._decode(tokens, prev_state, encodings[:, t])
                    token = probs.argmax()

                    if token != self._blank_token_idx:
                        prev_state = state
                        tokens.append(int(token))
                        emitted_tokens += 1
                    else:
                        break

            results.append(self._decode_tokens(tokens))

        return results
