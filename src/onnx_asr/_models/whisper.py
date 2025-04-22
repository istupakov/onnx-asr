from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

from onnx_asr._preprocessors.preprocessor import Preprocessor
from onnx_asr.asr import Asr


def bytes_to_unicode():
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


class Whisper(Asr):
    def __init__(self, model_parts: dict[str, Path], **kwargs):
        self._preprocessor = (
            Preprocessor("whisper128", **kwargs) if "v3" in str(model_parts["model"]) else Preprocessor("whisper80", **kwargs)
        )
        self._model = rt.InferenceSession(model_parts["model"], **kwargs)

        with Path(model_parts["vocab"]).open("rt") as f:
            self._tokens = {token: int(id) for token, id in (line.strip("\n").split(" ") for line in f.readlines())}
        self._vocab = {id: token for token, id in self._tokens.items()}
        self._byte_decoder = {v: k for k, v in bytes_to_unicode().items()}

    @staticmethod
    def _get_model_files(version: str | None = None) -> dict[str, str]:
        return {"vocab": "vocab.txt", "model": "whisper-*_beamsearch.onnx", "model-data": "whisper-*_beamsearch.onnx.data"}

    def _preprocess(self, waveforms: list[npt.NDArray[np.float32]]) -> npt.NDArray[np.float32]:
        chunk_length = 30
        input_length = chunk_length * 16_000

        def resize(waveform):
            if waveform.size < input_length:
                return np.pad(waveform, (0, input_length - waveform.size))
            else:
                return waveform[:input_length]

        input_features, _ = self._preprocessor(
            np.stack([resize(waveform) for waveform in waveforms]), np.repeat(input_length, len(waveforms))
        )
        return input_features

    def _postprocess(self, sequence: npt.NDArray[np.int32]) -> str:
        text = "".join(token for id in sequence if not (token := self._vocab[id]).startswith("<|"))
        return bytearray([self._byte_decoder[c] for c in text]).decode("utf-8", errors="replace").removeprefix(" ")

    def _process(
        self,
        input_features: npt.NDArray[np.float32],
        decoder_input_ids: list[list[int]],
        max_length=448,
        min_length=0,
        num_beams=1,
        num_return_sequences=1,
        length_penalty=1.0,
        repetition_penalty=1.0,
    ):
        (sequences,) = self._model.run(
            ["sequences"],
            {
                "input_features": input_features,
                "max_length": [max_length],
                "min_length": [min_length],
                "num_beams": [num_beams],
                "num_return_sequences": [num_return_sequences],
                "length_penalty": [length_penalty],
                "repetition_penalty": [repetition_penalty],
                "decoder_input_ids": decoder_input_ids,
            },
        )
        return sequences

    def _detect_language(self, input_features: npt.NDArray[np.float32]) -> list[int]:
        sequences = self._process(input_features, [[self._tokens["<|startoftranscript|>"]]] * len(input_features), max_length=3)
        return sequences[:, 0, 1]

    def _recognize_batch(self, waveforms: list[npt.NDArray[np.float32]], language: str | None = None) -> list[str]:
        input_features = self._preprocess(waveforms)
        languages = [self._tokens[f"<|{language}|>"]] * len(waveforms) if language else self._detect_language(input_features)

        sequences = self._process(
            input_features, [[self._tokens["<|startoftranscript|>"], lang, self._tokens["<|transcribe|>"]] for lang in languages]
        )
        return [self._postprocess(sequence) for sequence in sequences[:, 0, :]]
