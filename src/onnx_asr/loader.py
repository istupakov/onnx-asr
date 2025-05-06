"""Loader for ASR models."""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import onnxruntime as rt

from .adapters import TextResultsAsrAdapter
from .models import (
    GigaamV2Ctc,
    GigaamV2Rnnt,
    KaldiTransducer,
    NemoConformerCtc,
    NemoConformerRnnt,
    PyAnnoteVad,
    SileroVad,
    WhisperHf,
    WhisperOrt,
)
from .preprocessors import Resampler
from .vad import Vad

ModelNames = Literal[
    "gigaam-v2-ctc",
    "gigaam-v2-rnnt",
    "nemo-fastconformer-ru-ctc",
    "nemo-fastconformer-ru-rnnt",
    "nemo-parakeet-ctc-0.6b",
    "nemo-parakeet-rnnt-0.6b",
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


def _download_model(repo_id: str, files: list[str]) -> str:
    from huggingface_hub import snapshot_download

    files = [*files, *(str(path.with_suffix(".onnx?data")) for file in files if (path := Path(file)).suffix == ".onnx")]
    return snapshot_download(repo_id, allow_patterns=files)


def _find_files(path: str | Path | None, repo_id: str | None, files: dict[str, str]) -> dict[str, Path]:
    if path is None:
        if repo_id is None:
            raise NoModelNameOrPathSpecifiedError()
        path = _download_model(repo_id, list(files.values()))

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
) -> TextResultsAsrAdapter:
    """Load ASR model.

    Args:
        model: Model name or type (download from Hugging Face supported if full model name is provided):
                GigaAM v2 (`gigaam-v2-ctc` | `gigaam-v2-rnnt`),
                Kaldi Transducer (`kaldi-rnnt`)
                NeMo Conformer (`nemo-conformer-ctc` | `nemo-conformer-rnnt`)
                NeMo FastConformer Hybrid Large Ru P&C (`nemo-fastconformer-ru-ctc` | `nemo-fastconformer-ru-rnnt`)
                NeMo Parakeet 0.6B En (`nemo-parakeet-ctc-0.6b` | `nemo-parakeet-rnnt-0.6b`)
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
    repo_id: str | None = None
    match model:
        case "gigaam-v2-ctc":
            model_type = GigaamV2Ctc
            repo_id = "istupakov/gigaam-v2-onnx"
        case "gigaam-v2-rnnt":
            model_type = GigaamV2Rnnt
            repo_id = "istupakov/gigaam-v2-onnx"
        case "kaldi-rnnt" | "vosk":
            model_type = KaldiTransducer
        case "alphacep/vosk-model-ru" | "alphacep/vosk-model-small-ru":
            model_type = KaldiTransducer
            repo_id = model
        case "nemo-conformer-ctc":
            model_type = NemoConformerCtc
        case "nemo-fastconformer-ru-ctc":
            model_type = NemoConformerCtc
            repo_id = "istupakov/stt_ru_fastconformer_hybrid_large_pc_onnx"
        case "nemo-parakeet-ctc-0.6b":
            model_type = NemoConformerCtc
            repo_id = "istupakov/parakeet-ctc-0.6b-onnx"
        case "nemo-conformer-rnnt":
            model_type = NemoConformerRnnt
        case "nemo-fastconformer-ru-rnnt":
            model_type = NemoConformerRnnt
            repo_id = "istupakov/stt_ru_fastconformer_hybrid_large_pc_onnx"
        case "nemo-parakeet-rnnt-0.6b":
            model_type = NemoConformerRnnt
            repo_id = "istupakov/parakeet-rnnt-0.6b-onnx"
        case "whisper-ort":
            model_type = WhisperOrt
        case "whisper-base":
            model_type = WhisperOrt
            repo_id = "istupakov/whisper-base-onnx"
        case "whisper-hf":
            model_type = WhisperHf
        case model if model.startswith("onnx-community/") and "whisper" in model:
            model_type = WhisperHf
            repo_id = model
        case _:
            raise ModelNotSupportedError(model)

    if providers is None:
        providers = rt.get_available_providers()

    return TextResultsAsrAdapter(
        model_type(
            _find_files(path, repo_id, model_type._get_model_files(quantization)),
            sess_options=sess_options,
            providers=providers,
            provider_options=provider_options,
        ),
        Resampler(sess_options=sess_options, providers=providers, provider_options=provider_options),
    )


def load_vad(
    model: Literal["silero"] = "silero",
    path: str | Path | None = None,
    *,
    quantization: str | None = None,
    sess_options: rt.SessionOptions | None = None,
    providers: Sequence[str | tuple[str, dict]] | None = None,
    provider_options: Sequence[dict] | None = None,
) -> Vad:
    """Load VAD model.

    Args:
        model: VAD model name (supports download from Hugging Face).
        path: Path to directory with model files.
        quantization: Model quantization (`None` | `int8` | ... ).
        sess_options: Optional SessionOptions for onnxruntime.
        providers: Optional providers for onnxruntime.
        provider_options: Optional provider_options for onnxruntime.

    Returns:
        VAD model class.

    """
    model_type: type[SileroVad | PyAnnoteVad]
    match model:
        case "silero":
            model_type = SileroVad
            repo_id = "onnx-community/silero-vad"
        case "pyannote":
            model_type = PyAnnoteVad
            repo_id = "onnx-community/pyannote-segmentation-3.0"
        case _:
            raise ModelNotSupportedError(model)

    if providers is None:
        providers = rt.get_available_providers()

    return model_type(
        _find_files(path, repo_id, model_type._get_model_files(quantization)),
        sess_options=sess_options,
        providers=providers,
        provider_options=provider_options,
    )
