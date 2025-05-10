"""Loader for ASR models."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, get_args

import onnxruntime as rt

from onnx_asr.utils import OnnxSessionOptions

from .adapters import TextResultsAsrAdapter
from .models import (
    GigaamV2Ctc,
    GigaamV2Rnnt,
    KaldiTransducer,
    NemoConformerCtc,
    NemoConformerRnnt,
    NemoConformerTdt,
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
    "nemo-parakeet-tdt-0.6b-v2",
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
    "nemo-conformer-tdt",
    "vosk",
    "whisper-ort",
    "whisper",
]
VadNames = Literal["silero"]


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


class InvalidModelTypeInConfigError(Exception):
    """Invalid model type in config error."""

    def __init__(self, model_type: str) -> None:
        """Create error."""
        super().__init__(f"Invalid model type '{model_type}' in config.json.")


def _download_config(repo_id: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id, "config.json")


def _download_model(repo_id: str, files: list[str]) -> str:
    from huggingface_hub import snapshot_download

    files = [
        "config.json",
        *files,
        *(str(path.with_suffix(".onnx?data")) for file in files if (path := Path(file)).suffix == ".onnx"),
    ]
    return snapshot_download(repo_id, allow_patterns=files)


def _find_files(path: str | Path | None, repo_id: str | None, files: dict[str, str]) -> dict[str, Path]:
    if path is None:
        if repo_id is None:
            raise NoModelNameOrPathSpecifiedError()
        path = _download_model(repo_id, list(files.values()))

    if not Path(path).is_dir():
        raise ModelPathNotFoundError(path)

    if Path(path, "config.json").exists():
        files |= {"config": "config.json"}

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
    providers: Sequence[str | tuple[str, dict[Any, Any]]] | None = None,
    provider_options: Sequence[dict[Any, Any]] | None = None,
    cpu_preprocessing: bool = False,
) -> TextResultsAsrAdapter:
    """Load ASR model.

    Args:
        model: Model name or type (download from Hugging Face supported if full model name is provided):
                GigaAM v2 (`gigaam-v2-ctc` | `gigaam-v2-rnnt`),
                Kaldi Transducer (`kaldi-rnnt`)
                NeMo Conformer (`nemo-conformer-ctc` | `nemo-conformer-rnnt` | `nemo-conformer-tdt`)
                NeMo FastConformer Hybrid Large Ru P&C (`nemo-fastconformer-ru-ctc` | `nemo-fastconformer-ru-rnnt`)
                NeMo Parakeet 0.6B En (`nemo-parakeet-ctc-0.6b` | `nemo-parakeet-rnnt-0.6b` | `nemo-parakeet-tdt-0.6b-v2`)
                Vosk (`vosk` | `alphacep/vosk-model-ru` | `alphacep/vosk-model-small-ru`)
                Whisper Base exported with onnxruntime (`whisper-ort` | `whisper-base-ort`)
                Whisper from onnx-community (`whisper` | `onnx-community/whisper-large-v3-turbo` | `onnx-community/*whisper*`)
        path: Path to directory with model files.
        quantization: Model quantization (`None` | `int8` | ... ).
        sess_options: Optional SessionOptions for onnxruntime.
        providers: Optional providers for onnxruntime.
        provider_options: Optional provider_options for onnxruntime.
        cpu_preprocessing: Run preprocessors in CPU.

    Returns:
        ASR model class.

    """
    repo_id: str | None = None
    if "/" in model and path is None and not model.startswith("alphacep/"):
        repo_id = model
        with Path(_download_config(repo_id)).open("rt", encoding="utf-8") as f:
            config = json.load(f)
            config_model_type = config.get("model_type")
            if config_model_type in get_args(ModelTypes):
                model = config_model_type
            else:
                raise InvalidModelTypeInConfigError(config_model_type)

    model_type: type[GigaamV2Ctc | GigaamV2Rnnt | KaldiTransducer | NemoConformerCtc | NemoConformerRnnt | WhisperOrt | WhisperHf]
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
        case "nemo-conformer-tdt":
            model_type = NemoConformerTdt
        case "nemo-parakeet-tdt-0.6b-v2":
            model_type = NemoConformerTdt
            repo_id = "istupakov/parakeet-tdt-0.6b-v2-onnx"
        case "whisper-ort":
            model_type = WhisperOrt
        case "whisper-base":
            model_type = WhisperOrt
            repo_id = "istupakov/whisper-base-onnx"
        case "whisper":
            model_type = WhisperHf
        case _:
            raise ModelNotSupportedError(model)

    onnx_options: OnnxSessionOptions = {
        "sess_options": sess_options,
        "providers": providers or rt.get_available_providers(),
        "provider_options": provider_options,
        "cpu_preprocessing": cpu_preprocessing,
    }

    return TextResultsAsrAdapter(
        model_type(_find_files(path, repo_id, model_type._get_model_files(quantization)), onnx_options),
        Resampler(onnx_options),
    )


def load_vad(
    model: VadNames = "silero",
    path: str | Path | None = None,
    *,
    quantization: str | None = None,
    sess_options: rt.SessionOptions | None = None,
    providers: Sequence[str | tuple[str, dict[Any, Any]]] | None = None,
    provider_options: Sequence[dict[Any, Any]] | None = None,
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

    onnx_options: OnnxSessionOptions = {
        "sess_options": sess_options,
        "providers": providers or rt.get_available_providers(),
        "provider_options": provider_options,
    }

    return model_type(_find_files(path, repo_id, model_type._get_model_files(quantization)), onnx_options)
