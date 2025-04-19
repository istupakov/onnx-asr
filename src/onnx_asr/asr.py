"""Base ASR classes."""

import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ._preprocessors import Preprocessor
from .utils import pad_list, read_wav


class Asr(ABC):
    """Abstract ASR class with common interface and methods."""

    DECODE_SPACE_PATTERN = re.compile(r"\A\u2581|\u2581\B|(\u2581)\b")

    def __init__(self, preprocessor_name: Preprocessor.PreprocessorNames, vocab_path: Path):  # noqa: D107
        self._preprocessor = Preprocessor(preprocessor_name)
        self._vocab = dict(np.genfromtxt(vocab_path, dtype=None, delimiter=" ", usecols=[1, 0], encoding=None).tolist())
        self._blank_idx = next(key for (key, value) in self._vocab.items() if value == "<blk>")

    @staticmethod
    @abstractmethod
    def _get_model_parts() -> dict[str, str]:
        pass

    @abstractmethod
    def _encode(
        self, features: npt.NDArray[np.float32], features_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        pass

    @abstractmethod
    def _greedy_search(
        self, encoder_out: npt.NDArray[np.float32], encoder_out_lens: npt.NDArray[np.int64]
    ) -> Iterator[list[int]]:
        pass

    def _decode_tokens(self, tokens: list[int]) -> str:
        text = "".join([self._vocab[i] for i in tokens])
        return re.sub(self.DECODE_SPACE_PATTERN, lambda x: " " if x.group(1) else "", text)

    def _load_files(self, waveforms: list[npt.NDArray[np.float32] | str]) -> list[npt.NDArray[np.float32]]:
        for i in range(len(waveforms)):
            if isinstance(waveforms[i], str):
                waveform, sample_rate = read_wav(waveforms[i])  # type: ignore
                assert sample_rate == 16000, "Supported only 16 kHz sample rate"
                assert waveform.shape[1] == 1, "Supported only mono audio"
                waveforms[i] = waveform[:, 0]
            else:
                assert waveforms[i].ndim == 1, "waveform must be 1d numpy array"

        return waveforms  # type: ignore

    def _recognize_batch(self, waveforms: list[npt.NDArray[np.float32] | str]) -> list[str]:
        return list(
            map(
                self._decode_tokens,
                self._greedy_search(*self._encode(*self._preprocessor(*pad_list(self._load_files(waveforms))))),
            )
        )

    def recognize(self, waveform: str | npt.NDArray[np.float32] | list[str | npt.NDArray[np.float32]]) -> str | list[str]:
        """Recognize speech (single or batch).

        Args:
            waveform: Path to wav file (PCM_U8, PCM_16, PCM_24 and PCM_32 formats with 16 kHz sample rate are supported)
                      or Numpy array with PCM waveform.
                      A list of file paths or numpy arrays for batch recognition are also supported.

        Returns:
            Speech recognition results (single string or list for batch recognition).

        """
        if isinstance(waveform, list):
            return self._recognize_batch(waveform)
        return self._recognize_batch([waveform])[0]


class _CtcAsr(Asr):
    def _greedy_search(
        self, encoder_out: npt.NDArray[np.float32], encoder_out_lens: npt.NDArray[np.int64]
    ) -> Iterator[list[int]]:
        for log_probs, len in zip(encoder_out, encoder_out_lens, strict=True):
            tokens = log_probs[:len].argmax(axis=-1)
            tokens = tokens[np.diff(tokens).nonzero()]
            tokens = tokens[tokens != self._blank_idx]
            yield tokens


class _RnntAsr(Asr):
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

    def _greedy_search(
        self, encoder_out: npt.NDArray[np.float32], encoder_out_lens: npt.NDArray[np.int64]
    ) -> Iterator[list[int]]:
        for encodings, len in zip(encoder_out, encoder_out_lens, strict=True):
            prev_state = self._create_state()
            tokens = []

            for t in range(len):
                emitted_tokens = 0
                while emitted_tokens < self._max_tokens_per_step:
                    probs, state = self._decode(tokens, prev_state, encodings[:, t])
                    token = probs.argmax()

                    if token != self._blank_idx:
                        prev_state = state
                        tokens.append(int(token))
                        emitted_tokens += 1
                    else:
                        break

            yield tokens
