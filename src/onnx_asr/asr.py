"""Base ASR classes."""

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .preprocessors import Preprocessor
from .utils import pad_list, read_wav_files


class Asr(ABC):
    """Abstract ASR class with common interface and methods."""

    @staticmethod
    @abstractmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]: ...

    @abstractmethod
    def _recognize_batch(self, waveforms: list[npt.NDArray[np.float32]], language: str | None = None) -> list[str]: ...

    def recognize(
        self, waveform: str | npt.NDArray[np.float32] | list[str | npt.NDArray[np.float32]], language: str | None = None
    ) -> str | list[str]:
        """Recognize speech (single or batch).

        Args:
            waveform: Path to wav file (only PCM_U8, PCM_16, PCM_24 and PCM_32 formats with 16 kHz sample rate are supported)
                      or Numpy array with PCM waveform.
                      A list of file paths or numpy arrays for batch recognition are also supported.
            language: Speech language (only for Whisper models).

        Returns:
            Speech recognition results (single string or list for batch recognition).

        """
        if isinstance(waveform, list):
            return self._recognize_batch(read_wav_files(waveform), language)
        return self._recognize_batch(read_wav_files([waveform]), language)[0]


class _AsrWithDecoding(Asr):
    DECODE_SPACE_PATTERN = re.compile(r"\A\u2581|\u2581\B|(\u2581)\b")

    def __init__(self, preprocessor_name: str, vocab_path: Path, **kwargs: Any):
        self._preprocessor = Preprocessor(preprocessor_name, **kwargs)
        with Path(vocab_path).open("rt") as f:
            tokens = {token: int(id) for token, id in (line.strip("\n").split(" ") for line in f.readlines())}
        self._vocab = {id: token for token, id in tokens.items()}
        self._blank_idx = tokens["<blk>"]

    @abstractmethod
    def _encode(
        self, features: npt.NDArray[np.float32], features_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]: ...

    @abstractmethod
    def _decoding(self, encoder_out: npt.NDArray[np.float32], encoder_out_lens: npt.NDArray[np.int64]) -> Iterable[list[int]]: ...

    def _decode_tokens(self, tokens: list[int]) -> str:
        text = "".join([self._vocab[i] for i in tokens])
        return re.sub(self.DECODE_SPACE_PATTERN, lambda x: " " if x.group(1) else "", text)

    def _recognize_batch(self, waveforms: list[npt.NDArray[np.float32]], language: str | None = None) -> list[str]:
        return list(map(self._decode_tokens, self._decoding(*self._encode(*self._preprocessor(*pad_list(waveforms))))))


class _AsrWithCtcDecoding(_AsrWithDecoding):
    def _decoding(self, encoder_out: npt.NDArray[np.float32], encoder_out_lens: npt.NDArray[np.int64]) -> Iterable[list[int]]:
        assert encoder_out.shape[-1] <= len(self._vocab)

        for log_probs, log_probs_len in zip(encoder_out, encoder_out_lens, strict=True):
            tokens = log_probs[:log_probs_len].argmax(axis=-1)
            tokens = tokens[np.diff(tokens).nonzero()]
            tokens = tokens[tokens != self._blank_idx]
            yield tokens


class _AsrWithRnntDecoding(_AsrWithDecoding):
    @abstractmethod
    def _create_state(self) -> tuple: ...

    @property
    @abstractmethod
    def _max_tokens_per_step(self) -> int: ...

    @abstractmethod
    def _decode(
        self, prev_tokens: list[int], prev_state: tuple, encoder_out: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], tuple]: ...

    def _decoding(self, encoder_out: npt.NDArray[np.float32], encoder_out_lens: npt.NDArray[np.int64]) -> Iterable[list[int]]:
        for encodings, encodings_len in zip(encoder_out, encoder_out_lens, strict=True):
            prev_state = self._create_state()
            tokens: list[int] = []

            for t in range(encodings_len):
                emitted_tokens = 0
                while emitted_tokens < self._max_tokens_per_step:
                    probs, state = self._decode(tokens, prev_state, encodings[:, t])
                    assert probs.shape[-1] <= len(self._vocab)

                    token = probs.argmax()

                    if token != self._blank_idx:
                        prev_state = state
                        tokens.append(int(token))
                        emitted_tokens += 1
                    else:
                        break

            yield tokens
