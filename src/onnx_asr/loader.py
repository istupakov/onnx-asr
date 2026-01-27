"""Loader for ASR models."""

import json
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Generic, Literal, Protocol, TypeAlias, TypeVar

import onnxruntime as rt

from onnx_asr.adapters import TextResultsAsrAdapter
from onnx_asr.asr import Asr, Preprocessor
from onnx_asr.models.gigaam import GigaamV2Ctc, GigaamV2Rnnt, GigaamV3E2eCtc, GigaamV3E2eRnnt
from onnx_asr.models.kaldi import KaldiTransducer
from onnx_asr.models.nemo import NemoConformerAED, NemoConformerCtc, NemoConformerRnnt, NemoConformerTdt
from onnx_asr.models.pyannote import PyAnnoteVad
from onnx_asr.models.silero import SileroVad
from onnx_asr.models.tone import TOneCtc
from onnx_asr.models.whisper import WhisperHf, WhisperOrt
from onnx_asr.onnx import OnnxSessionOptions, get_onnx_providers, update_onnx_providers
from onnx_asr.preprocessors.preprocessor import ConcurrentPreprocessor, IdentityPreprocessor, OnnxPreprocessor
from onnx_asr.preprocessors.resampler import Resampler
from onnx_asr.utils import (
    InvalidModelTypeInConfigError,
    ModelFileNotFoundError,
    ModelNotSupportedError,
    ModelPathNotDirectoryError,
    MoreThanOneModelFileFoundError,
    NoModelNameOrPathSpecifiedError,
)

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
"""Supported ASR model names (can be automatically downloaded from the Hugging Face)."""

ModelTypes = Literal[
    "kaldi-rnnt",
    "nemo-conformer-ctc",
    "nemo-conformer-rnnt",
    "nemo-conformer-tdt",
    "nemo-conformer-aed",
    "t-one-ctc",
    "vosk",
    "whisper-ort",
    "whisper",
]
"""Supported ASR model types."""

VadNames = Literal["silero"]
"""Supported VAD model names (can be automatically downloaded from the Hugging Face)."""


class PreprocessorRuntimeConfig(OnnxSessionOptions, total=False):
    """Preprocessor runtime config."""

    max_concurrent_workers: int | None
    """Max parallel preprocessing threads (None - auto, 1 - without parallel processing)."""


class _Model(Protocol):
    @staticmethod
    def _get_excluded_providers() -> list[str]: ...

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]: ...


T = TypeVar("T", bound=_Model)


class _Loader(ABC, Generic[T]):
    offline: bool = False
    local_dir: Path | None = None
    repo_id: str | None = None

    def __init__(self, model: str, path: str | Path | None = None, *, offline: bool | None = None):  # noqa: C901
        if path is not None:
            self.local_dir = Path(path)
            if self.local_dir.exists():
                self.offline = True
                if not self.local_dir.is_dir():
                    raise ModelPathNotDirectoryError(self.local_dir)

        if offline is not None:
            self.offline = offline

        if "/" in model:
            self.repo_id = model
        elif model in self._model_repos:
            self.repo_id = self._model_repos[model]
        elif not (self.offline and self.local_dir):
            raise NoModelNameOrPathSpecifiedError

        if model in self._model_types:
            self._model_type = self._model_types[model]
        elif "/" in model:
            with self.resolve_config().open("rt", encoding="utf-8") as f:
                config = json.load(f)

            config_model_type: str = config.get("model_type")
            if "/" in config_model_type or config_model_type not in self._model_types:
                raise InvalidModelTypeInConfigError(config_model_type)
            self._model_type = self._model_types[config_model_type]
        else:
            raise ModelNotSupportedError(model)

    def get_excluded_providers(self) -> list[str]:
        return self._model_type._get_excluded_providers()

    @property
    @abstractmethod
    def _model_repos(self) -> dict[str, str]: ...

    @property
    @abstractmethod
    def _model_types(self) -> dict[str, type[T]]: ...

    def _download_config(self, *, local_files_only: bool) -> Path:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        assert self.repo_id is not None
        return Path(
            hf_hub_download(self.repo_id, "config.json", local_dir=self.local_dir, local_files_only=local_files_only)
        )  # nosec

    def _download_model(self, quantization: str | None, *, local_files_only: bool) -> Path:
        from huggingface_hub import snapshot_download  # noqa: PLC0415

        files = list(self._model_type._get_model_files(quantization).values())
        files = [
            "config.json",
            *files,
            *(str(path.with_suffix(".onnx?data")) for file in files if (path := Path(file)).suffix == ".onnx"),
        ]
        assert self.repo_id is not None
        return Path(
            snapshot_download(
                self.repo_id, local_dir=self.local_dir, local_files_only=local_files_only, allow_patterns=files
            )  # nosec
        )

    def _resolve_model_files(self, path: Path, quantization: str | None) -> dict[str, Path]:
        files = self._model_type._get_model_files(quantization)
        if Path(path, "config.json").exists():
            files |= {"config": "config.json"}

        def find(filename: str) -> Path:
            files = list(path.glob(filename))
            if len(files) > 1:
                raise MoreThanOneModelFileFoundError(filename, path)
            if len(files) == 0 or not files[0].is_file():
                raise ModelFileNotFoundError(filename, path)
            return files[0]

        return {key: find(filename) for key, filename in files.items()}

    def resolve_config(self) -> Path:
        if self.offline and self.local_dir:
            config_path = Path(self.local_dir, "config.json")
            if not config_path.is_file():
                raise ModelFileNotFoundError(config_path.name, self.local_dir)
            return config_path

        try:
            return self._download_config(local_files_only=True)
        except FileNotFoundError:
            if self.offline:
                raise
            return self._download_config(local_files_only=False)

    def resolve_model(self, *, quantization: str | None = None) -> dict[str, Path]:
        if self.offline and self.local_dir:
            return self._resolve_model_files(self.local_dir, quantization)

        try:
            return self._resolve_model_files(self._download_model(quantization, local_files_only=True), quantization)
        except FileNotFoundError:
            if self.offline:
                raise
            return self._resolve_model_files(self._download_model(quantization, local_files_only=False), quantization)


class AsrLoader(_Loader[Asr]):
    """Loader class for ASR models."""

    @property
    def _model_repos(self) -> dict[str, str]:
        return {
            "gigaam-v2-ctc": "istupakov/gigaam-v2-onnx",
            "gigaam-v2-rnnt": "istupakov/gigaam-v2-onnx",
            "gigaam-v3-ctc": "istupakov/gigaam-v3-onnx",
            "gigaam-v3-rnnt": "istupakov/gigaam-v3-onnx",
            "gigaam-v3-e2e-ctc": "istupakov/gigaam-v3-onnx",
            "gigaam-v3-e2e-rnnt": "istupakov/gigaam-v3-onnx",
            "nemo-fastconformer-ru-ctc": "istupakov/stt_ru_fastconformer_hybrid_large_pc_onnx",
            "nemo-fastconformer-ru-rnnt": "istupakov/stt_ru_fastconformer_hybrid_large_pc_onnx",
            "nemo-parakeet-ctc-0.6b": "istupakov/parakeet-ctc-0.6b-onnx",
            "nemo-parakeet-rnnt-0.6b": "istupakov/parakeet-rnnt-0.6b-onnx",
            "nemo-parakeet-tdt-0.6b-v2": "istupakov/parakeet-tdt-0.6b-v2-onnx",
            "nemo-parakeet-tdt-0.6b-v3": "istupakov/parakeet-tdt-0.6b-v3-onnx",
            "nemo-canary-1b-v2": "istupakov/canary-1b-v2-onnx",
            "whisper-base": "istupakov/whisper-base-onnx",
        }

    @property
    def _model_types(self) -> dict[str, type[Asr]]:
        return {
            "gigaam-v2-ctc": GigaamV2Ctc,
            "gigaam-v2-rnnt": GigaamV2Rnnt,
            "gigaam-v3-ctc": GigaamV2Ctc,
            "gigaam-v3-rnnt": GigaamV2Rnnt,
            "gigaam-v3-e2e-ctc": GigaamV3E2eCtc,
            "gigaam-v3-e2e-rnnt": GigaamV3E2eRnnt,
            "nemo-fastconformer-ru-ctc": NemoConformerCtc,
            "nemo-fastconformer-ru-rnnt": NemoConformerRnnt,
            "nemo-parakeet-ctc-0.6b": NemoConformerCtc,
            "nemo-parakeet-rnnt-0.6b": NemoConformerRnnt,
            "nemo-parakeet-tdt-0.6b-v2": NemoConformerTdt,
            "nemo-parakeet-tdt-0.6b-v3": NemoConformerTdt,
            "nemo-canary-1b-v2": NemoConformerAED,
            "whisper-base": WhisperOrt,
            "kaldi-rnnt": KaldiTransducer,
            "nemo-conformer-ctc": NemoConformerCtc,
            "nemo-conformer-rnnt": NemoConformerRnnt,
            "nemo-conformer-tdt": NemoConformerTdt,
            "nemo-conformer-aed": NemoConformerAED,
            "t-one-ctc": TOneCtc,
            "vosk": KaldiTransducer,
            "whisper-ort": WhisperOrt,
            "whisper": WhisperHf,
            "alphacep/vosk-model-ru": KaldiTransducer,
            "alphacep/vosk-model-small-ru": KaldiTransducer,
            "t-tech/t-one": TOneCtc,
        }

    def create_model(
        self,
        asr_config: OnnxSessionOptions,
        preprocessor_config: PreprocessorRuntimeConfig,
        resampler_config: OnnxSessionOptions,
        *,
        quantization: str | None = None,
    ) -> TextResultsAsrAdapter:
        """Create ASR model."""

        def create_preprocessor(name: str) -> Preprocessor:
            if name == "identity":
                return IdentityPreprocessor()

            providers = get_onnx_providers(preprocessor_config)
            if name == "kaldi" and providers and providers != ["CPUExecutionProvider"]:
                name = "kaldi_fast"

            max_concurrent_workers = preprocessor_config.pop("max_concurrent_workers", 1)
            preprocessor = OnnxPreprocessor(name, preprocessor_config)
            if max_concurrent_workers == 1:
                return preprocessor
            return ConcurrentPreprocessor(preprocessor, max_concurrent_workers)

        return TextResultsAsrAdapter(
            self._model_type(self.resolve_model(quantization=quantization), create_preprocessor, asr_config),
            Resampler(self._model_type._get_sample_rate(), resampler_config),
        )


Vad: TypeAlias = SileroVad | PyAnnoteVad


class VadLoader(_Loader[Vad]):
    """Loader class for VAD models."""

    @property
    def _model_repos(self) -> dict[str, str]:
        return {"silero": "onnx-community/silero-vad", "pyannote": "onnx-community/pyannote-segmentation-3.0"}

    @property
    def _model_types(self) -> dict[str, type[Vad]]:
        return {"silero": SileroVad, "pyannote": PyAnnoteVad}

    def create_model(self, config: OnnxSessionOptions, *, quantization: str | None = None) -> Vad:
        """Create VAD model."""
        return self._model_type(self.resolve_model(quantization=quantization), config)


def load_model(
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
                GigaAM v3 (`gigaam-v3-ctc` | `gigaam-v3-rnnt` |
                           `gigaam-v3-e2e-ctc` | `gigaam-v3-e2e-rnnt`)
                Kaldi Transducer (`kaldi-rnnt`)
                NeMo Conformer (`nemo-conformer-ctc` | `nemo-conformer-rnnt` | `nemo-conformer-tdt` |
                                `nemo-conformer-aed`)
                NeMo FastConformer Hybrid Large Ru P&C (`nemo-fastconformer-ru-ctc` |
                                                        `nemo-fastconformer-ru-rnnt`)
                NeMo Parakeet 0.6B En (`nemo-parakeet-ctc-0.6b` | `nemo-parakeet-rnnt-0.6b` |
                                       `nemo-parakeet-tdt-0.6b-v2`)
                NeMo Parakeet 0.6B Multilingual (`nemo-parakeet-tdt-0.6b-v3`)
                NeMo Canary (`nemo-canary-1b-v2`)
                T-One (`t-one-ctc` | `t-tech/t-one`)
                Vosk (`vosk` | `alphacep/vosk-model-ru` | `alphacep/vosk-model-small-ru`)
                Whisper Base exported with onnxruntime (`whisper-ort` | `whisper-base-ort`)
                Whisper from onnx-community (`whisper` | `onnx-community/whisper-large-v3-turbo` |
                                             `onnx-community/*whisper*`)
        path: Path to directory with model files.
        quantization: Model quantization (`None` | `int8` | ... ).
        sess_options: Default SessionOptions for onnxruntime.
        providers: Default providers for onnxruntime.
        provider_options: Default provider_options for onnxruntime.
        cpu_preprocessing: Deprecated and ignored, use `preprocessor_config` and `resampler_config` instead.
        asr_config: ASR ONNX config.
        preprocessor_config: Preprocessor ONNX and concurrency config.
        resampler_config: Resampler ONNX config.

    Returns:
        ASR model class.

    Raises:
        utils.ModelLoadingError: Model loading error (onnx-asr specific).

    """
    if cpu_preprocessing is not None:
        warnings.warn(
            "The cpu_preprocessing argument is deprecated and ignored (use preprocessor_config and resampler_config).",
            stacklevel=2,
        )

    loader = AsrLoader(model, path)

    default_onnx_config: OnnxSessionOptions = {
        "sess_options": sess_options,
        "providers": providers or rt.get_available_providers(),
        "provider_options": provider_options,
    }

    if asr_config is None:
        asr_config = update_onnx_providers(default_onnx_config, excluded_providers=loader.get_excluded_providers())

    if preprocessor_config is None:
        preprocessor_config = {
            **update_onnx_providers(
                default_onnx_config,
                new_options={"TensorrtExecutionProvider": {"trt_fp16_enable": False, "trt_int8_enable": False}},
                excluded_providers=OnnxPreprocessor._get_excluded_providers(),
            ),
            "max_concurrent_workers": 1,
        }

    if resampler_config is None:
        resampler_config = update_onnx_providers(
            default_onnx_config, excluded_providers=Resampler._get_excluded_providers()
        )

    return loader.create_model(asr_config, preprocessor_config, resampler_config, quantization=quantization)


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

    Raises:
        utils.ModelLoadingError: Model loading error (onnx-asr specific).

    """
    loader = VadLoader(model, path)

    onnx_options = update_onnx_providers(
        {"providers": rt.get_available_providers()}, excluded_providers=loader.get_excluded_providers()
    ) | {
        "sess_options": sess_options,
        "providers": providers,
        "provider_options": provider_options,
    }

    return loader.create_model(onnx_options, quantization=quantization)
