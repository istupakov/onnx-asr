import re
import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
from typing import Iterator, Any
from .utils import read_wav, pad_list


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
    def _recognize(self, waveforms: npt.NDArray[np.float32], waveforms_lens: npt.NDArray[np.int64]) -> Iterator[list[int]]:
        pass

    def _decode_tokens(self, tokens: list[int]) -> str:
        text = "".join([self._vocabulary[i] for i in tokens])
        return re.sub(r"\A\u2581|\u2581\B", "", text).replace("\u2581", " ")

    def recognize(self, waveform: str | npt.NDArray[np.float32]) -> str:
        return self.recognize_batch([waveform])[0]

    def recognize_batch(self, waveforms: list[str | npt.NDArray[np.float32]]) -> list[str]:
        for i in range(len(waveforms)):
            if isinstance(waveforms[i], str):
                waveform, sample_rate = read_wav(waveforms[i])  # type: ignore
                assert sample_rate == 16000, "Supported only 16 kHz sample rate"
                assert waveform.shape[1] == 1, "Supported only mono audio"
                waveforms[i] = waveform[:, 0]

        return list(map(self._decode_tokens, self._recognize(*pad_list(waveforms))))  # type: ignore


class CtcAsr(Asr):
    def _recognize(self, waveforms: npt.NDArray[np.float32], waveforms_lens: npt.NDArray[np.int64]) -> Iterator[list[int]]:
        for log_probs, len in zip(*self._encode(waveforms, waveforms_lens)):
            tokens = log_probs[:len].argmax(axis=-1)
            tokens = tokens[np.diff(tokens).nonzero()]
            tokens = tokens[tokens != self._blank_token_idx]
            yield tokens


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

    def _recognize(self, waveforms: npt.NDArray[np.float32], waveforms_lens: npt.NDArray[np.int64]) -> Iterator[list[int]]:
        for encodings, len in zip(*self._encode(waveforms, waveforms_lens)):
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

            yield tokens
