"""NeMo model implementations."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

from onnx_asr.asr import _AsrWithCtcDecoding, _AsrWithDecoding, _AsrWithTransducerDecoding
from onnx_asr.utils import OnnxSessionOptions, is_float32_array, is_int64_array


class _NemoConformer(_AsrWithDecoding):
    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        return {"vocab": "vocab.txt"}

    @property
    def _preprocessor_name(self) -> str:
        return f"nemo{self.config.get('features_size', 80)}"

    @property
    def _subsampling_factor(self) -> int:
        return self.config.get("subsampling_factor", 8)


class NemoConformerCtc(_AsrWithCtcDecoding, _NemoConformer):
    """NeMo Conformer CTC model implementations."""

    def __init__(self, model_files: dict[str, Path], onnx_options: OnnxSessionOptions):
        """Create NeMo Conformer CTC model.

        Args:
            model_files: Dict with paths to model files.
            onnx_options: Options for onnxruntime InferenceSession.

        """
        super().__init__(model_files, onnx_options)
        self._model = rt.InferenceSession(model_files["model"], **onnx_options)

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        return {"model": f"model{suffix}.onnx"} | _NemoConformer._get_model_files(quantization)

    def _encode(
        self, features: npt.NDArray[np.float32], features_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        (logprobs,) = self._model.run(["logprobs"], {"audio_signal": features, "length": features_lens})
        assert is_float32_array(logprobs)
        return logprobs, (features_lens - 1) // self._subsampling_factor + 1


_STATE_TYPE = tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]


class NemoConformerRnnt(_AsrWithTransducerDecoding[_STATE_TYPE], _NemoConformer):
    """NeMo Conformer RNN-T model implementations."""

    def __init__(self, model_files: dict[str, Path], onnx_options: OnnxSessionOptions):
        """Create NeMo Conformer RNN-T model.

        Args:
            model_files: Dict with paths to model files.
            onnx_options: Options for onnxruntime InferenceSession.

        """
        super().__init__(model_files, onnx_options)
        self._encoder = rt.InferenceSession(model_files["encoder"], **onnx_options)
        self._decoder_joint = rt.InferenceSession(model_files["decoder_joint"], **onnx_options)

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        return {
            "encoder": f"encoder-model{suffix}.onnx",
            "decoder_joint": f"decoder_joint-model{suffix}.onnx",
        } | _NemoConformer._get_model_files(quantization)

    @property
    def _max_tokens_per_step(self) -> int:
        return self.config.get("max_tokens_per_step", 10)

    def _encode(
        self, features: npt.NDArray[np.float32], features_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        encoder_out, encoder_out_lens = self._encoder.run(
            ["outputs", "encoded_lengths"], {"audio_signal": features, "length": features_lens}
        )
        assert is_float32_array(encoder_out)
        assert is_int64_array(encoder_out_lens)
        return encoder_out.transpose(0, 2, 1), encoder_out_lens

    def _create_state(self) -> _STATE_TYPE:
        shapes = {x.name: x.shape for x in self._decoder_joint.get_inputs()}
        return (
            np.zeros(shape=(shapes["input_states_1"][0], 1, shapes["input_states_1"][2]), dtype=np.float32),
            np.zeros(shape=(shapes["input_states_2"][0], 1, shapes["input_states_2"][2]), dtype=np.float32),
        )

    def _decode(
        self, prev_tokens: list[int], prev_state: _STATE_TYPE, encoder_out: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], int, _STATE_TYPE]:
        outputs, state1, state2 = self._decoder_joint.run(
            ["outputs", "output_states_1", "output_states_2"],
            {
                "encoder_outputs": encoder_out[None, :, None],
                "targets": [[prev_tokens[-1] if prev_tokens else self._blank_idx]],
                "target_length": [1],
                "input_states_1": prev_state[0],
                "input_states_2": prev_state[1],
            },
        )
        assert is_float32_array(outputs)
        assert is_float32_array(state1)
        assert is_float32_array(state2)
        return np.squeeze(outputs), -1, (state1, state2)


class NemoConformerTdt(NemoConformerRnnt):
    """NeMo Conformer TDT model implementations."""

    def _decode(
        self, prev_tokens: list[int], prev_state: _STATE_TYPE, encoder_out: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], int, _STATE_TYPE]:
        output, _, state = super()._decode(prev_tokens, prev_state, encoder_out)
        return output[: self._vocab_size], int(output[self._vocab_size :].argmax()), state


class NemoConformerAED(_NemoConformer):
    """NeMo Conformer AED model implementations."""

    def __init__(self, model_files: dict[str, Path], onnx_options: OnnxSessionOptions):
        """Create NeMo Conformer AED model.

        Args:
            model_files: Dict with paths to model files.
            onnx_options: Options for onnxruntime InferenceSession.

        """
        super().__init__(model_files, onnx_options)
        self._encoder = rt.InferenceSession(model_files["encoder"], **onnx_options)
        self._decoder = rt.InferenceSession(model_files["decoder"], **onnx_options)

        self._tokens = {token: id for id, token in self._vocab.items()}
        self._eos_token_id = self._tokens["<|endoftext|>"]
        self._transcribe_input = np.array(
            [
                [
                    self._tokens["<|startofcontext|>"],
                    self._tokens["<|startoftranscript|>"],
                    self._tokens["<|emo:undefined|>"],
                    self._tokens["<|en|>"],
                    self._tokens["<|en|>"],
                    self._tokens["<|pnc|>"],
                    self._tokens["<|noitn|>"],
                    self._tokens["<|notimestamp|>"],
                    self._tokens["<|nodiarize|>"],
                ]
            ],
            dtype=np.int64,
        )

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        return {
            "encoder": f"encoder-model{suffix}.onnx",
            "decoder": f"decoder-model{suffix}.onnx",
        } | _NemoConformer._get_model_files(quantization)

    @property
    def _max_sequence_length(self) -> int:
        return self.config.get("max_sequence_length", 1024)

    def _encode(
        self, features: npt.NDArray[np.float32], features_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        encoder_embeddings, encoder_mask = self._encoder.run(
            ["encoder_embeddings", "encoder_mask"], {"audio_signal": features, "length": features_lens}
        )
        assert is_float32_array(encoder_embeddings)
        assert is_int64_array(encoder_mask)
        return encoder_embeddings, encoder_mask

    def _decode(
        self,
        input_ids: npt.NDArray[np.int64],
        encoder_embeddings: npt.NDArray[np.float32],
        encoder_mask: npt.NDArray[np.int64],
        decoder_mems: npt.NDArray[np.float32],
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        logits, decoder_hidden_states = self._decoder.run(
            ["logits", "decoder_hidden_states"],
            {
                "input_ids": input_ids if decoder_mems.shape[2] == 0 else input_ids[:, -1:],
                "encoder_embeddings": encoder_embeddings,
                "encoder_mask": encoder_mask,
                "decoder_mems": decoder_mems,
            },
        )
        assert is_float32_array(logits)
        assert is_float32_array(decoder_hidden_states)
        return logits, decoder_hidden_states

    def _decoding(
        self, encoder_embeddings: npt.NDArray[np.float32], encoder_mask: npt.NDArray[np.int64], language: str | None
    ) -> Iterator[tuple[list[int], list[int]]]:
        batch_size = encoder_embeddings.shape[0]
        tokens = np.repeat(self._transcribe_input, batch_size, axis=0)

        if language:
            tokens[:, 3] = self._tokens[f"<|{language}|>"]
            tokens[:, 4] = self._tokens[f"<|{language}|>"]

        shapes = {x.name: x.shape for x in self._decoder.get_inputs()}
        decoder_mems = np.empty((shapes["decoder_mems"][0], batch_size, 0, shapes["decoder_mems"][3]), dtype=np.float32)
        while tokens.shape[1] < self._max_sequence_length:
            logits, decoder_mems = self._decode(tokens, encoder_embeddings, encoder_mask, decoder_mems)

            next_tokens = np.argmax(logits[:, -1], axis=-1)
            if (next_tokens == self._eos_token_id).all():
                break

            tokens = np.concatenate((tokens, next_tokens[:, None]), axis=-1)

        return (([id for id in tok if not self._vocab[id].startswith("<|")], []) for tok in tokens)
