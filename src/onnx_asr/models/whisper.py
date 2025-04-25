"""Whisper model implementations."""

import json
import typing
from abc import abstractmethod
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

from onnx_asr.asr import Asr
from onnx_asr.preprocessors.preprocessor import Preprocessor


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
    def __init__(self, model_files: dict[str, Path], **kwargs: typing.Any):
        with model_files["preprocessor_config"].open() as f:
            preprocessor_config = json.load(f)
        assert preprocessor_config["feature_size"] in [80, 128], "feature_size not in [80, 128]"

        self._input_length = preprocessor_config["n_samples"]
        self._preprocessor = Preprocessor(f"whisper{preprocessor_config['feature_size']}", **kwargs)

        with model_files["vocab"].open() as f:
            self._tokens: dict[str, int] = json.load(f)

        with model_files["added_tokens"].open() as f:
            self._tokens |= json.load(f)

        self._vocab = {id: token for token, id in self._tokens.items()}
        self._bos_token_id = self._tokens["<|startoftranscript|>"]
        self._eos_token_id = self._tokens["<|endoftext|>"]
        self._byte_decoder = {v: k for k, v in bytes_to_unicode().items()}
        self._decoder_input = np.array(
            [
                [
                    self._bos_token_id,
                    self._eos_token_id,
                    self._tokens["<|transcribe|>"],
                    self._tokens["<|notimestamps|>"],
                ]
            ]
        )

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        return {
            "preprocessor_config": "preprocessor_config.json",
            "vocab": "vocab.json",
            "added_tokens": "added_tokens.json",
        }

    def _preprocess(self, waveforms: list[npt.NDArray[np.float32]]) -> npt.NDArray[np.float32]:
        def resize(waveform: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
            if waveform.size < self._input_length:
                return np.pad(waveform, (0, self._input_length - waveform.size))
            else:
                return waveform[: self._input_length]

        input_features, _ = self._preprocessor(
            np.stack([resize(waveform) for waveform in waveforms]), np.repeat(self._input_length, len(waveforms))
        )
        return input_features

    @abstractmethod
    def _decoding(self, input_features: npt.NDArray, tokens: npt.NDArray, max_length: int = 448) -> npt.NDArray: ...

    def _decode_tokens(self, tokens: npt.NDArray) -> str:
        text = "".join(token for id in tokens if (token := self._vocab[id]) and not token.startswith("<|"))
        return bytearray([self._byte_decoder[c] for c in text]).decode("utf-8", errors="replace").removeprefix(" ")

    def _recognize_batch(self, waveforms: list[npt.NDArray[np.float32]], language: str | None = None) -> list[str]:
        input_features = self._preprocess(waveforms)
        input_tokens = np.repeat(self._decoder_input, len(waveforms), axis=0)

        if language:
            input_tokens[:, 1] = self._tokens[f"<|{language}|>"]
        else:
            input_tokens_detect_lang = np.repeat([[self._bos_token_id]], len(waveforms), axis=0)
            input_tokens[:, 1] = self._decoding(input_features, input_tokens_detect_lang, 3)[:, 1]

        return list(map(self._decode_tokens, self._decoding(input_features, input_tokens)))


class WhisperOrt(_Whisper):
    """Whisper (exported with onnxruntime) model implementation."""

    def __init__(self, model_files: dict[str, Path], **kwargs: typing.Any):
        """Create Whisper model.

        Args:
            model_files: Dict with paths to model files.
            kwargs: Additional parameters for onnxruntime.InferenceSession.

        """
        super().__init__(model_files, **kwargs)
        self._model = rt.InferenceSession(model_files["model"], **kwargs)

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        return {"model": f"whisper-*_beamsearch{suffix}.onnx"} | _Whisper._get_model_files(quantization)

    def _decoding(self, input_features: npt.NDArray, tokens: npt.NDArray, max_length: int = 448) -> npt.NDArray:
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
        return typing.cast(npt.NDArray, sequences[:, 0, :])


class WhisperHf(_Whisper):
    """Whisper (exported with optimum) model implementation."""

    def __init__(self, model_files: dict[str, Path], **kwargs: typing.Any):
        """Create Whisper model.

        Args:
            model_files: Dict with paths to model files.
            kwargs: Additional parameters for onnxruntime.InferenceSession.

        """
        super().__init__(model_files, **kwargs)
        self._encoder = rt.InferenceSession(model_files["encoder"], **kwargs)
        self._decoder = rt.InferenceSession(model_files["decoder"], **kwargs)

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        return {
            "encoder": f"**/encoder_model{suffix}.onnx",
            "decoder": f"**/decoder_model{suffix}.onnx",
        } | _Whisper._get_model_files(suffix)

    def _preprocess(self, waveforms: list[npt.NDArray[np.float32]]) -> npt.NDArray[np.float32]:
        input_features = super()._preprocess(waveforms)
        (last_hidden_state,) = self._encoder.run(
            ["last_hidden_state"],
            {"input_features": input_features},
        )
        return typing.cast(npt.NDArray[np.float32], last_hidden_state)

    def _decode(self, tokens: npt.NDArray, encoder_out: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        (logits,) = self._decoder.run(["logits"], {"input_ids": tokens, "encoder_hidden_states": encoder_out})
        return typing.cast(npt.NDArray[np.float32], logits)

    def _decoding(self, input_features: npt.NDArray[np.float32], tokens: npt.NDArray, max_length: int = 448) -> npt.NDArray:
        for _ in range(tokens.shape[-1], max_length):
            logits = self._decode(tokens, input_features)
            next_tokens = logits[:, -1].argmax(axis=-1)
            next_tokens[tokens[:, -1] == self._eos_token_id] = self._eos_token_id
            tokens = np.hstack((tokens, next_tokens[:, None]))
            if (tokens[:, -1] == self._eos_token_id).all():
                break

        return tokens
