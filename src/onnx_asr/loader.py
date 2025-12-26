"""Loader for ASR models."""

import json
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, get_args

import onnxruntime as rt

from onnx_asr.adapters import TextResultsAsrAdapter
from onnx_asr.asr import AsrRuntimeConfig
from onnx_asr.models import (
    GigaamV2Ctc,
    GigaamV2Rnnt,
    GigaamV3E2eCtc,
    GigaamV3E2eRnnt,
    KaldiTransducer,
    NemoConformerAED,
    NemoConformerCtc,
    NemoConformerRnnt,
    NemoConformerTdt,
    PyAnnoteVad,
    SileroVad,
    TOneCtc,
    WhisperHf,
    WhisperOrt,
)
from onnx_asr.onnx import OnnxSessionOptions, update_onnx_providers
from onnx_asr.preprocessors import PreprocessorRuntimeConfig, Resampler
from onnx_asr.preprocessors.preprocessor import Preprocessor
from onnx_asr.vad import Vad

ModelNames = Literal[
    "gigaam-v2-ctc",
    "gigaam-v2-rnnt",
    "gigaam-v3-ctc",
    "gigaam-v3-rnnt",
    "gigaam-v3-e2e-ctc",
    "gigaam-v3-e2e-rnnt",
    "nemo-fastconformer-ru-ctc",
    "nemo-fastconformer-ru-rnnt",
    "nemo-parakeet-ctc-0.6b",
    "nemo-parakeet-rnnt-0.6b",
    "nemo-parakeet-tdt-0.6b-v2",
    "nemo-parakeet-tdt-0.6b-v3",
    "nemo-canary-1b-v2",
    "alphacep/vosk-model-ru",
    "alphacep/vosk-model-small-ru",
    "t-tech/t-one",
    "whisper-base",
]
ModelTypes = Literal[
    "gigaam-v2-ctc",
    "gigaam-v2-rnnt",
    "gigaam-v3-e2e-ctc",
    "gigaam-v3-e2e-rnnt",
    "kaldi-rnnt",
    "nemo-conformer-ctc",
    "nemo-conformer-rnnt",
    "nemo-conformer-tdt",
    "nemo-conformer-aed",
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


class ModelPathNotDirectoryError(NotADirectoryError):
    """Model path not a directory error."""

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
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    return hf_hub_download(repo_id, "config.json")


def _download_model(repo_id: str, files: list[str], *, local_dir: str | Path | None, local_files_only: bool) -> str:
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    files = [
        "config.json",
        *files,
        *(str(path.with_suffix(".onnx?data")) for file in files if (path := Path(file)).suffix == ".onnx"),
    ]
    return snapshot_download(repo_id, local_dir=local_dir, local_files_only=local_files_only, allow_patterns=files)


def _find_files(path: str | Path, files: dict[str, str]) -> dict[str, Path]:
    if not Path(path).is_dir():
        raise ModelPathNotDirectoryError(path)

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


def _download_files(repo_id: str | None, path: str | Path | None, files: dict[str, str]) -> dict[str, Path]:
    if path is not None and Path(path).exists():
        return _find_files(path, files)

    if repo_id is None:
        raise NoModelNameOrPathSpecifiedError

    try:
        return _find_files(_download_model(repo_id, list(files.values()), local_dir=path, local_files_only=True), files)
    except (FileNotFoundError, ModelFileNotFoundError):
        return _find_files(_download_model(repo_id, list(files.values()), local_dir=path, local_files_only=False), files)


def _find_model_type(repo_id: str, path: str | Path | None) -> str:
    if path is not None and Path(path).exists():
        config_path = Path(path, "config.json")
        if not config_path.is_file():
            raise ModelFileNotFoundError(config_path.name, path)
    else:
        config_path = Path(_download_config(repo_id))

    with config_path.open("rt", encoding="utf-8") as f:
        config = json.load(f)
        config_model_type: str = config.get("model_type")
        if config_model_type not in get_args(ModelTypes):
            raise InvalidModelTypeInConfigError(config_model_type)

        return config_model_type


def load_model(  # noqa: C901
    model: str | ModelNames | ModelTypes,
    path: str | Path | None = None,
    *,
    quantization: str | None = None,
    sess_options: rt.SessionOptions | None = None,
    providers: Sequence[str | tuple[str, dict[Any, Any]]] | None = None,
    provider_options: Sequence[dict[Any, Any]] | None = None,
    cpu_preprocessing: bool | None = None,
    asr_config: OnnxSessionOptions | None = None,
    preprocessor_config: PreprocessorRuntimeConfig | None = None,
    resampler_config: OnnxSessionOptions | None = None,
) -> TextResultsAsrAdapter:
    """Load ASR model.

    Args:
        model: Model name or type (download from Hugging Face supported if full model name is provided):
                GigaAM v2 (`gigaam-v2-ctc` | `gigaam-v2-rnnt`)
                GigaAM v3 (`gigaam-v3-ctc` | `gigaam-v3-rnnt` | `gigaam-v3-e2e-ctc` | `gigaam-v3-e2e-rnnt`)
                Kaldi Transducer (`kaldi-rnnt`)
                NeMo Conformer (`nemo-conformer-ctc` | `nemo-conformer-rnnt` | `nemo-conformer-tdt` | `nemo-conformer-aed`)
                NeMo FastConformer Hybrid Large Ru P&C (`nemo-fastconformer-ru-ctc` | `nemo-fastconformer-ru-rnnt`)
                NeMo Parakeet 0.6B En (`nemo-parakeet-ctc-0.6b` | `nemo-parakeet-rnnt-0.6b` | `nemo-parakeet-tdt-0.6b-v2`)
                NeMo Parakeet 0.6B Multilingual (`nemo-parakeet-tdt-0.6b-v3`)
                NeMo Canary (`nemo-canary-1b-v2`)
                T-One (`t-tech/t-one`)
                Vosk (`vosk` | `alphacep/vosk-model-ru` | `alphacep/vosk-model-small-ru`)
                Whisper Base exported with onnxruntime (`whisper-ort` | `whisper-base-ort`)
                Whisper from onnx-community (`whisper` | `onnx-community/whisper-large-v3-turbo` | `onnx-community/*whisper*`)
        path: Path to directory with model files.
        quantization: Model quantization (`None` | `int8` | ... ).
        sess_options: Default SessionOptions for onnxruntime.
        providers: Default providers for onnxruntime.
        provider_options: Default provider_options for onnxruntime.
        cpu_preprocessing: Deprecated and ignored, use preprocessor_config and resampler_config instead.
        asr_config: ASR ONNX config.
        preprocessor_config: Preprocessor ONNX and concurrency config.
        resampler_config: Resampler ONNX config.

    Returns:
        ASR model class.

    """
    if cpu_preprocessing is not None:
        warnings.warn(
            "The cpu_preprocessing argument is deprecated and ignored (use preprocessor_config and resampler_config instead).",
            stacklevel=2,
        )

    if "/" in model and not model.startswith(("alphacep/", "t-tech/")):
        repo_id = model
        model = _find_model_type(repo_id, path)
    else:
        repo_id = None

    model_type: type[
        GigaamV2Ctc
        | GigaamV2Rnnt
        | KaldiTransducer
        | NemoConformerCtc
        | NemoConformerRnnt
        | NemoConformerAED
        | TOneCtc
        | WhisperOrt
        | WhisperHf
    ]
    default_repo_id = None
    match model:
        case "gigaam-v2-ctc":
            model_type = GigaamV2Ctc
            default_repo_id = "istupakov/gigaam-v2-onnx"
        case "gigaam-v2-rnnt":
            model_type = GigaamV2Rnnt
            default_repo_id = "istupakov/gigaam-v2-onnx"
        case "gigaam-v3-ctc":
            model_type = GigaamV2Ctc
            default_repo_id = "istupakov/gigaam-v3-onnx"
        case "gigaam-v3-rnnt":
            model_type = GigaamV2Rnnt
            default_repo_id = "istupakov/gigaam-v3-onnx"
        case "gigaam-v3-e2e-ctc":
            model_type = GigaamV3E2eCtc
            default_repo_id = "istupakov/gigaam-v3-onnx"
        case "gigaam-v3-e2e-rnnt":
            model_type = GigaamV3E2eRnnt
            default_repo_id = "istupakov/gigaam-v3-onnx"
        case "kaldi-rnnt" | "vosk":
            model_type = KaldiTransducer
        case "alphacep/vosk-model-ru" | "alphacep/vosk-model-small-ru":
            model_type = KaldiTransducer
            default_repo_id = model
        case "nemo-conformer-ctc":
            model_type = NemoConformerCtc
        case "nemo-fastconformer-ru-ctc":
            model_type = NemoConformerCtc
            default_repo_id = "istupakov/stt_ru_fastconformer_hybrid_large_pc_onnx"
        case "nemo-parakeet-ctc-0.6b":
            model_type = NemoConformerCtc
            default_repo_id = "istupakov/parakeet-ctc-0.6b-onnx"
        case "nemo-conformer-rnnt":
            model_type = NemoConformerRnnt
        case "nemo-fastconformer-ru-rnnt":
            model_type = NemoConformerRnnt
            default_repo_id = "istupakov/stt_ru_fastconformer_hybrid_large_pc_onnx"
        case "nemo-parakeet-rnnt-0.6b":
            model_type = NemoConformerRnnt
            default_repo_id = "istupakov/parakeet-rnnt-0.6b-onnx"
        case "nemo-conformer-tdt":
            model_type = NemoConformerTdt
        case "nemo-parakeet-tdt-0.6b-v2":
            model_type = NemoConformerTdt
            default_repo_id = "istupakov/parakeet-tdt-0.6b-v2-onnx"
        case "nemo-parakeet-tdt-0.6b-v3":
            model_type = NemoConformerTdt
            default_repo_id = "istupakov/parakeet-tdt-0.6b-v3-onnx"
        case "nemo-conformer-aed":
            model_type = NemoConformerAED
        case "nemo-canary-1b-v2":
            model_type = NemoConformerAED
            default_repo_id = "istupakov/canary-1b-v2-onnx"
        case "t-tech/t-one":
            model_type = TOneCtc
            default_repo_id = model
        case "whisper-ort":
            model_type = WhisperOrt
        case "whisper-base":
            model_type = WhisperOrt
            default_repo_id = "istupakov/whisper-base-onnx"
        case "whisper":
            model_type = WhisperHf
        case _:
            raise ModelNotSupportedError(model)

    default_onnx_config: OnnxSessionOptions = {
        "sess_options": sess_options,
        "providers": providers or rt.get_available_providers(),
        "provider_options": provider_options,
    }

    if asr_config is None:
        asr_config = update_onnx_providers(default_onnx_config, excluded_providers=model_type._get_excluded_providers())

    if preprocessor_config is None:
        preprocessor_config = {
            **update_onnx_providers(
                default_onnx_config,
                new_options={"TensorrtExecutionProvider": {"trt_fp16_enable": False, "trt_int8_enable": False}},
                excluded_providers=Preprocessor._get_excluded_providers(),
            ),
            "max_concurrent_workers": 1,
        }

    if resampler_config is None:
        resampler_config = update_onnx_providers(default_onnx_config, excluded_providers=Resampler._get_excluded_providers())

    return TextResultsAsrAdapter(
        model_type(
            _download_files(repo_id or default_repo_id, path, model_type._get_model_files(quantization)),
            AsrRuntimeConfig(asr_config, preprocessor_config),
        ),
        Resampler(model_type._get_sample_rate(), resampler_config),
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

    onnx_options = update_onnx_providers(
        {"providers": rt.get_available_providers()}, excluded_providers=model_type._get_excluded_providers()
    ) | {
        "sess_options": sess_options,
        "providers": providers,
        "provider_options": provider_options,
    }

    return model_type(_download_files(repo_id, path, model_type._get_model_files(quantization)), onnx_options)
