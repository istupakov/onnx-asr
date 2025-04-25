"""Loader for ASR models."""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, get_args

import onnxruntime as rt

from .asr import Asr
from .models import (
    GigaamV2Ctc,
    GigaamV2Rnnt,
    KaldiTransducer,
    NemoConformerCtc,
    NemoConformerRnnt,
    WhisperHf,
    WhisperOrt,
)

ModelNames = Literal[
    "gigaam-v2-ctc",
    "gigaam-v2-rnnt",
    "nemo-fastconformer-ru-ctc",
    "nemo-fastconformer-ru-rnnt",
    "alphacep/vosk-model-ru",
    "alphacep/vosk-model-small-ru",
    "whisper-base",
]
ModelTypes = Literal[
    "gigaam-v2-ctc",
    "gigaam-v2-rnnt",
    "kaldi-rnnt",
    "nemo-conformer-ctc",
    "nemo-conformer-rnnt",
    "vosk",
    "whisper-ort",
    "whisper-hf",
]
ModelVersions = Literal["int8"] | None


def _get_model_class(
    model: str,
) -> (
    type[GigaamV2Ctc]
    | type[GigaamV2Rnnt]
    | type[KaldiTransducer]
    | type[NemoConformerCtc]
    | type[NemoConformerRnnt]
    | type[WhisperOrt]
    | type[WhisperHf]
):
    match model.split("/"):
        case ("gigaam-v2-ctc",):
            return GigaamV2Ctc
        case ("gigaam-v2-rnnt",):
            return GigaamV2Rnnt
        case ("kaldi-rnnt" | "vosk",) | ("alphacep", "vosk-model-ru" | "vosk-model-small-ru"):
            return KaldiTransducer
        case ("nemo-conformer-ctc" | "nemo-fastconformer-ru-ctc",):
            return NemoConformerCtc
        case ("nemo-conformer-rnnt" | "nemo-fastconformer-ru-rnnt",):
            return NemoConformerRnnt
        case ("whisper-ort" | "whisper-base",):
            return WhisperOrt
        case ("whisper-hf",):
            return WhisperHf
        case ("onnx-community", name) if "whisper" in name:
            return WhisperHf
        case _:
            raise ValueError(f"Model '{model}' not supported!")  # noqa: TRY003


def _resolve_paths(path: str | Path, model_files: dict[str, str]) -> dict[str, Path]:
    assert Path(path).is_dir(), f"The path '{path}' is not a directory."

    def find(filename: str) -> Path:
        files = list(Path(path).glob(filename))
        assert len(files) > 0, f"File '{filename}' not found in path '{path}'."
        assert len(files) == 1, f"Found more than 1 file '{filename}' found in path '{path}'."
        return files[0]

    return {key: find(filename) for key, filename in model_files.items()}


def _download_model(model: str, files: list[str]) -> str:
    from huggingface_hub import snapshot_download

    match model:
        case "gigaam-v2-ctc" | "gigaam-v2-rnnt":
            repo_id = "istupakov/gigaam-v2-onnx"
        case "nemo-fastconformer-ru-ctc" | "nemo-fastconformer-ru-rnnt":
            repo_id = "istupakov/stt_ru_fastconformer_hybrid_large_pc_onnx"
        case "whisper-base":
            repo_id = "istupakov/whisper-base-onnx"
        case _:
            repo_id = model

    files = [*files, *(str(path.with_suffix(".onnx?data")) for file in files if (path := Path(file)).suffix == ".onnx")]
    return snapshot_download(repo_id, allow_patterns=files)


def load_model(
    model: str | ModelNames | ModelTypes,
    path: str | Path | None = None,
    quantization: str | None = None,
    providers: Sequence[str | tuple[str, dict]] | None = None,
) -> Asr:
    """Load ASR model.

    Args:
        model: Model name or type (specific models support downloading from Hugging Face):
                GigaAM v2 (`gigaam-v2-ctc` | `gigaam-v2-rnnt`),
                Kaldi Transducer (`kaldi-rnnt`)
                NeMo Conformer (`nemo-conformer-ctc` | `nemo-conformer-rnnt`)
                NeMo FastConformer Hybrid Large Ru P&C (`nemo-fastconformer-ru-ctc` | `nemo-fastconformer-ru-rnnt`)
                Vosk (`vosk` | `alphacep/vosk-model-ru` | `alphacep/vosk-model-small-ru`)
                Whisper Base exported with onnxruntime (`whisper-ort` | `whisper-base-ort`)
                Whisper from onnx-community (`whisper-hf` | `onnx-community/whisper-large-v3-turbo` | `onnx-community/*whisper*`)
        path: Path to directory with model files.
        quantization: Model quantization (`None` | `int8` | ... ).
        providers: Optional providers for onnxruntime.

    Returns:
        ASR model class.

    """
    model_class = _get_model_class(model)
    files = model_class._get_model_files(quantization)

    if path is None:
        assert model in get_args(ModelNames) or model.startswith("onnx-community/"), (
            "If the path is not specified, you must specify a specific model name."
        )
        path = _download_model(model, list(files.values()))

    if providers is None:
        providers = rt.get_available_providers()

    return model_class(_resolve_paths(path, files), providers=providers)
