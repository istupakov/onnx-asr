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


class ModelNotSupportedError(ValueError):
    """Model not supported error."""

    def __init__(self, model: str):
        """Create error."""
        super().__init__(f"Model '{model}' not supported!")


class ModelPathNotFoundError(NotADirectoryError):
    """Model path not found error."""

    def __init__(self, path: str | Path):
        """Create error."""
        super().__init__(f"The path '{path}' is not a directory.")


class ModelFileNotFoundError(FileNotFoundError):
    """Model file not found error."""

    def __init__(self, filename: str | Path, path: str | Path):
        """Create error."""
        super().__init__(f"File '{filename}' not found in path '{path}'.")


class MoreThanOneModelFileFoundError(Exception):
    """More than one model file found error."""

    def __init__(self, filename: str | Path, path: str | Path):
        """Create error."""
        super().__init__(f"Found more than 1 file '{filename}' found in path '{path}'.")


class NoModelNameOrPathSpecifiedError(Exception):
    """No model name or path specified error."""

    def __init__(self) -> None:
        """Create error."""
        super().__init__("If the path is not specified, you must specify a specific model name.")


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


def _find_files(model: str, path: str | Path | None, files: dict[str, str]) -> dict[str, Path]:
    if path is None:
        if not (model in get_args(ModelNames) or model.startswith("onnx-community/")):
            raise NoModelNameOrPathSpecifiedError()
        path = _download_model(model, list(files.values()))

    if not Path(path).is_dir():
        raise ModelPathNotFoundError(path)

    def find(filename: str) -> Path:
        files = list(Path(path).glob(filename))
        if len(files) == 0:
            raise ModelFileNotFoundError(filename, path)
        if len(files) > 1:
            raise MoreThanOneModelFileFoundError(filename, path)
        return files[0]

    return {key: find(filename) for key, filename in files.items()}


def load_model(
    model: str | ModelNames | ModelTypes,
    path: str | Path | None = None,
    *,
    quantization: str | None = None,
    sess_options: rt.SessionOptions | None = None,
    providers: Sequence[str | tuple[str, dict]] | None = None,
    provider_options: Sequence[dict] | None = None,
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
        sess_options: Optional SessionOptions for onnxruntime.
        providers: Optional providers for onnxruntime.
        provider_options: Optional provider_options for onnxruntime.

    Returns:
        ASR model class.

    """
    model_type: type[GigaamV2Ctc | GigaamV2Rnnt | KaldiTransducer | NemoConformerCtc | NemoConformerRnnt | WhisperOrt | WhisperHf]
    match model.split("/"):
        case ("gigaam-v2-ctc",):
            model_type = GigaamV2Ctc
        case ("gigaam-v2-rnnt",):
            model_type = GigaamV2Rnnt
        case ("kaldi-rnnt" | "vosk",) | ("alphacep", "vosk-model-ru" | "vosk-model-small-ru"):
            model_type = KaldiTransducer
        case ("nemo-conformer-ctc" | "nemo-fastconformer-ru-ctc",):
            model_type = NemoConformerCtc
        case ("nemo-conformer-rnnt" | "nemo-fastconformer-ru-rnnt",):
            model_type = NemoConformerRnnt
        case ("whisper-ort" | "whisper-base",):
            model_type = WhisperOrt
        case ("whisper-hf",):
            model_type = WhisperHf
        case ("onnx-community", name) if "whisper" in name:
            model_type = WhisperHf
        case _:
            raise ModelNotSupportedError(model)

    if providers is None:
        providers = rt.get_available_providers()

    return model_type(
        _find_files(model, path, model_type._get_model_files(quantization)),
        sess_options=sess_options,
        providers=providers,
        provider_options=provider_options,
    )
