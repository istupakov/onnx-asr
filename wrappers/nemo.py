"""Wrapper for NeMo models."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import nemo.collections.asr as nemo_asr
import numpy as np
import numpy.typing as npt
import torch
from nemo.utils.nemo_logging import Logger

from onnx_asr.asr import TimestampedResult


class NemoASR:
    """Wrapper model for NeMo Toolkit ASR."""

    def __init__(self, model_name: str):
        """Create wrapper."""
        self.logger = Logger()
        self.logger.setLevel(Logger.ERROR)

        self.model: Any = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
        self.model.change_decoding_strategy({"strategy": "greedy_batch"})
        self.model.eval()

    @staticmethod
    def _get_sample_rate() -> Literal[8_000, 16_000]:
        return 16_000

    def recognize_batch(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64], /, **kwargs: object | None
    ) -> Iterator[TimestampedResult]:
        """Recognize waveforms batch."""
        language = kwargs.get("language")
        target_language = kwargs.get("target_language") or language
        pnc = kwargs.get("pnc")
        if pnc is not None:
            pnc = "yes" if pnc is True or pnc == "pnc" else "no"

        for waveform, waveform_len in zip(waveforms, waveforms_len, strict=True):
            hypot = self.model.transcribe(
                waveform[:waveform_len], verbose=False, source_lang=language, target_lang=target_language, pnc=pnc
            )
            yield TimestampedResult(hypot[0].text)

    def export(self, path: str | Path) -> None:
        """Export model to ONNX."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if isinstance(self.model, nemo_asr.models.EncDecMultiTaskModel):
            self.model.to("cpu")
            _CanaryEncoder(self.model).export().save(path / "encoder-model.onnx")
            _CanaryDecoder(self.model).export(path / "decoder-model.onnx")
            model_type = "nemo-conformer-aed"
        else:
            self.model.export(str(path / "model.onnx"))
            model_type = "nemo-conformer"

        with (path / "config.json").open("w") as f:
            json.dump({"model_type": model_type, "features_size": self.model.cfg.preprocessor.features}, f, indent=4)

        with (path / "vocab.txt").open("wt") as f:
            for i, token in enumerate([*self.model.tokenizer.vocab, "<blk>"]):
                f.write(f"{token} {i}\n")


class _CanaryEncoder(torch.nn.Module):
    def __init__(self, model: nemo_asr.models.EncDecMultiTaskModel):
        super().__init__()
        self.encoder = model.encoder
        self.encoder_decoder_proj = model.encoder_decoder_proj

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, encoded_len = self.encoder(audio_signal=audio_signal, length=length)
        encoder_embeddings = self.encoder_decoder_proj(encoded.permute(0, 2, 1))

        arange = torch.arange(encoder_embeddings.shape[1], device=encoded_len.device)
        encoder_mask = arange.expand(encoded_len.shape[0], encoder_embeddings.shape[1]) < encoded_len.unsqueeze(1)

        return encoder_embeddings, encoder_mask.to(torch.int64)

    def export(self) -> torch.onnx.ONNXProgram:
        batch = torch.export.Dim("batch", min=1)
        seq_len = torch.export.Dim("seq_len", min=9)
        return torch.onnx.export(
            self.eval(),
            self.encoder.input_example(2),
            input_names=["audio_signal", "length"],
            output_names=["encoder_embeddings", "encoder_mask"],
            dynamic_shapes={"audio_signal": {0: batch, 2: seq_len}, "length": {0: batch}},
            verify=True,
        )


class _CanaryDecoder(torch.nn.Module):
    def __init__(self, model: nemo_asr.models.EncDecMultiTaskModel):
        super().__init__()
        self.decoder = model.transf_decoder
        self.classifier = model.log_softmax
        self.eos = model.tokenizer.eos_id

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_embeddings: torch.Tensor,
        encoder_mask: torch.Tensor,
        decoder_mems: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        decoder_embeddings = self.decoder.embedding(input_ids=input_ids, start_pos=decoder_mems.shape[2])
        decoder_mask = (input_ids != self.eos).float()

        decoder_hidden_states, _ = self.decoder.decoder(
            decoder_states=decoder_embeddings,
            decoder_mask=decoder_mask,
            encoder_states=encoder_embeddings,
            encoder_mask=encoder_mask,
            decoder_mems_list=decoder_mems,
            return_mems=True,
            return_mems_as_list=False,
        )

        logits = self.classifier(hidden_states=decoder_hidden_states[-1, :, -1:])

        return logits, decoder_hidden_states

    def export(self, filename: str | Path) -> None:
        input_ids, _, encoder_embeddings, encoder_mask, decoder_mems = self.decoder.input_example(2)
        decoder_mems = torch.rand(
            (decoder_mems.shape[1], decoder_mems.shape[0], 0, decoder_mems.shape[-1]), device=decoder_mems.device
        )
        torch.onnx.export(
            self.eval(),
            (input_ids, encoder_embeddings, encoder_mask, decoder_mems),
            filename,
            input_names=["input_ids", "encoder_embeddings", "encoder_mask", "decoder_mems"],
            output_names=["logits", "decoder_hidden_states"],
            dynamo=False,
            dynamic_axes={
                "input_ids": {0: "batch", 1: "input_len"},
                "encoder_embeddings": {0: "batch", 1: "encoded_len"},
                "encoder_mask": {0: "batch", 1: "encoded_len"},
                "decoder_mems": {1: "batch", 2: "mems_len"},
            },
        )


class _CanaryDecoderDynamo(torch.nn.Module):
    def __init__(self, model: nemo_asr.models.EncDecMultiTaskModel):
        super().__init__()
        self.decoder = model.transf_decoder
        self.classifier = model.log_softmax
        self.eos = model.tokenizer.eos_id

    def decode_with_mems(
        self,
        input_ids: torch.Tensor,
        encoder_embeddings: torch.Tensor,
        encoder_mask: torch.Tensor,
        decoder_mems: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = input_ids[:, -1:]
        decoder_embeddings = self.decoder.embedding(input_ids=input_ids, start_pos=decoder_mems.shape[2])
        decoder_mask = (input_ids != self.eos).float()
        decoder_hidden_states, _ = self.decoder.decoder(
            decoder_states=decoder_embeddings,
            decoder_mask=decoder_mask,
            encoder_states=encoder_embeddings,
            encoder_mask=encoder_mask,
            decoder_mems_list=decoder_mems,
            return_mems=True,
            return_mems_as_list=False,
        )
        return decoder_hidden_states

    def decode_without_mems(
        self,
        input_ids: torch.Tensor,
        encoder_embeddings: torch.Tensor,
        encoder_mask: torch.Tensor,
        decoder_mems: torch.Tensor,
    ) -> torch.Tensor:
        decoder_embeddings = self.decoder.embedding(input_ids=input_ids, start_pos=0)
        decoder_mask = (input_ids != self.eos).float()
        decoder_hidden_states, _ = self.decoder.decoder(
            decoder_states=decoder_embeddings,
            decoder_mask=decoder_mask,
            encoder_states=encoder_embeddings,
            encoder_mask=encoder_mask,
            decoder_mems_list=None,
            return_mems=True,
            return_mems_as_list=False,
        )
        return decoder_hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_embeddings: torch.Tensor,
        encoder_mask: torch.Tensor,
        decoder_mems: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        decoder_hidden_states = torch.cond(
            decoder_mems.shape[2] > 0,
            self.decode_with_mems,
            self.decode_without_mems,
            (input_ids, encoder_embeddings, encoder_mask, decoder_mems),
        )

        logits = self.classifier(hidden_states=decoder_hidden_states[-1, :, -1:])
        return logits, decoder_hidden_states

    def export(self) -> torch.onnx.ONNXProgram:
        batch = torch.export.Dim("batch", min=1)
        encoded_len = torch.export.Dim("encoded_len", min=1)
        input_len = torch.export.Dim("input_len", min=1)
        mems_len = torch.export.Dim("mems_len", min=0)

        input_ids, _, encoder_embeddings, encoder_mask, decoder_mems = self.decoder.input_example(2)
        decoder_mems = torch.transpose(decoder_mems, 0, 1)

        return torch.onnx.export(
            self.eval(),
            (input_ids, encoder_embeddings, encoder_mask, decoder_mems[:, :, :0]),
            input_names=["input_ids", "encoder_embeddings", "encoder_mask", "decoder_mems"],
            output_names=["logits", "decoder_hidden_states"],
            dynamic_shapes={
                "input_ids": {0: batch, 1: input_len},
                "encoder_embeddings": {0: batch, 1: encoded_len},
                "encoder_mask": {0: batch, 1: encoded_len},
                "decoder_mems": {1: batch, 2: mems_len},
            },
            verify=True,
        )
