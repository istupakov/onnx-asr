"""Whisper model implementations."""

import json
import typing
from abc import abstractmethod
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

from onnx_asr.asr import Asr, TimestampedResult
from onnx_asr.utils import OnnxSessionOptions, is_float32_array, is_int32_array


@typing.no_type_check
def bytes_to_unicode() -> dict[int, str]:
    """Magic func copied from transformers.models.gpt2.tokenization_gpt2.bytes_to_unicode."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))  # noqa: B905


class _Whisper(Asr):
    def __init__(self, model_files: dict[str, Path], onnx_options: OnnxSessionOptions):
        super().__init__(model_files, onnx_options)

        with model_files["vocab"].open("rt", encoding="utf-8") as f:
            self._tokens: dict[str, int] = json.load(f)

        with model_files["added_tokens"].open("rt", encoding="utf-8") as f:
            self._tokens |= json.load(f)

        self._vocab = {id: token for token, id in self._tokens.items()}
        self._bos_token_id = self._tokens["<|startoftranscript|>"]
        self._eos_token_id = self._tokens["<|endoftext|>"]
        self._byte_decoder = {v: k for k, v in bytes_to_unicode().items()}
        self._transcribe_input = np.array(
            [
                [
                    self._bos_token_id,
                    self._eos_token_id,
                    self._tokens["<|transcribe|>"],
                    self._tokens["<|notimestamps|>"],
                ]
            ],
            dtype=np.int64,
        )
        self._detect_lang_input = np.array([[self._bos_token_id]], dtype=np.int64)

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        return {"vocab": "vocab.json", "added_tokens": "added_tokens.json"}

    def _encode(self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64]) -> npt.NDArray[np.float32]:
        input_features, _ = self._preprocessor(waveforms, waveforms_len)
        return input_features

    @abstractmethod
    def _decoding(
        self, input_features: npt.NDArray[np.float32], tokens: npt.NDArray[np.int64], max_length: int = 448
    ) -> npt.NDArray[np.int64]: ...

    def _decode_tokens(self, tokens: npt.NDArray[np.int64]) -> TimestampedResult:
        text = "".join(token for id in tokens if (token := self._vocab[id]) and not token.startswith("<|"))
        return TimestampedResult(
            bytearray([self._byte_decoder[c] for c in text]).decode("utf-8", errors="replace").removeprefix(" ")
        )

    def recognize_batch(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64], language: str | None
    ) -> Iterator[TimestampedResult]:
        input_encoding = self._encode(waveforms, waveforms_len)
        input_tokens = np.repeat(self._transcribe_input, len(waveforms), axis=0)

        if language:
            input_tokens[:, 1] = self._tokens[f"<|{language}|>"]
        else:
            input_tokens_detect_lang = np.repeat(self._detect_lang_input, len(waveforms), axis=0)
            input_tokens[:, 1] = self._decoding(input_encoding, input_tokens_detect_lang, 3)[:, 1]

        return map(self._decode_tokens, self._decoding(input_encoding, input_tokens))


class WhisperOrt(_Whisper):
    """Whisper (exported with onnxruntime) model implementation."""

    def __init__(self, model_files: dict[str, Path], onnx_options: OnnxSessionOptions):
        """Create Whisper model.

        Args:
            model_files: Dict with paths to model files.
            onnx_options: Options for onnxruntime InferenceSession.

        """
        super().__init__(model_files, onnx_options)
        self._model = rt.InferenceSession(model_files["model"], **onnx_options)

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        return {"model": f"whisper-*_beamsearch{suffix}.onnx"} | _Whisper._get_model_files(quantization)

    @property
    def _preprocessor_name(self) -> str:
        return f"whisper{self.config.get('features_size', 80)}"

    def _decoding(
        self, input_features: npt.NDArray[np.float32], tokens: npt.NDArray[np.int64], max_length: int = 448
    ) -> npt.NDArray[np.int64]:
        (sequences,) = self._model.run(
            ["sequences"],
            {
                "input_features": input_features,
                "max_length": [max_length],
                "min_length": [0],
                "num_beams": [1],
                "num_return_sequences": [1],
                "length_penalty": [1.0],
                "repetition_penalty": [1.0],
                "decoder_input_ids": tokens.astype(np.int32),
            },
        )
        assert is_int32_array(sequences)
        return sequences[:, 0, :].astype(np.int64)


class WhisperHf(_Whisper):
    """Whisper (exported with optimum) model implementation."""

    def __init__(self, model_files: dict[str, Path], onnx_options: OnnxSessionOptions):
        """Create Whisper model.

        Args:
            model_files: Dict with paths to model files.
            onnx_options: Options for onnxruntime InferenceSession.

        """
        super().__init__(model_files, onnx_options)
        self._encoder = rt.InferenceSession(model_files["encoder"], **onnx_options)
        self._decoder = rt.InferenceSession(model_files["decoder"], **onnx_options)

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        return {
            "encoder": f"**/encoder_model{suffix}.onnx",
            "decoder": f"**/decoder_model{suffix}.onnx",
        } | _Whisper._get_model_files(suffix)

    @property
    def _preprocessor_name(self) -> str:
        return f"whisper{self.config.get('num_mel_bins', 80)}"

    def _encode(self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64]) -> npt.NDArray[np.float32]:
        input_features = super()._encode(waveforms, waveforms_len)
        (last_hidden_state,) = self._encoder.run(["last_hidden_state"], {"input_features": input_features})
        assert is_float32_array(last_hidden_state)
        return last_hidden_state

    def _decode(self, tokens: npt.NDArray[np.int64], encoder_out: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        (logits,) = self._decoder.run(["logits"], {"input_ids": tokens, "encoder_hidden_states": encoder_out})
        assert is_float32_array(logits)
        return logits

    def _decoding(
        self, input_features: npt.NDArray[np.float32], tokens: npt.NDArray[np.int64], max_length: int = 448
    ) -> npt.NDArray[np.int64]:
        for _ in range(tokens.shape[-1], max_length):
            logits = self._decode(tokens, input_features)
            next_tokens = logits[:, -1].argmax(axis=-1)
            next_tokens[tokens[:, -1] == self._eos_token_id] = self._eos_token_id
            tokens = np.hstack((tokens, next_tokens[:, None]))
            if (tokens[:, -1] == self._eos_token_id).all():
                break

        return tokens
