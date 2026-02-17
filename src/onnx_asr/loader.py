"""Loader for ASR and VAD models."""

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, TypeAlias

import onnxruntime as rt

from onnx_asr.adapters import SeAdapter, TextResultsAsrAdapter
from onnx_asr.asr import Asr, Preprocessor
from onnx_asr.models.gigaam import GigaamV2Ctc, GigaamV2Rnnt, GigaamV3E2eCtc, GigaamV3E2eRnnt
from onnx_asr.models.kaldi import KaldiTransducer
from onnx_asr.models.nemo import NemoConformerAED, NemoConformerCtc, NemoConformerRnnt, NemoConformerTdt
from onnx_asr.models.pyannote import PyAnnoteVad
from onnx_asr.models.silero import SileroVad
from onnx_asr.models.tone import TOneCtc
from onnx_asr.models.wespeaker import WespeakerEmbeddings
from onnx_asr.models.whisper import WhisperHf, WhisperOrt
from onnx_asr.onnx import OnnxSessionOptions, get_onnx_providers, update_onnx_providers
from onnx_asr.preprocessors.numpy_preprocessor import (
    GigaamPreprocessorNumpy,
    KaldiPreprocessorNumpy,
    NemoPreprocessorNumpy,
    WhisperPreprocessorNumpy,
)
from onnx_asr.preprocessors.preprocessor import ConcurrentPreprocessor, IdentityPreprocessor, OnnxPreprocessor
from onnx_asr.preprocessors.resampler import Resampler
from onnx_asr.resolver import Resolver
from onnx_asr.se import SpeakerEmbedding
from onnx_asr.utils import (
    ModelNotSupportedError,
)
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

AsrTypes: TypeAlias = (
    GigaamV2Ctc
    | GigaamV2Rnnt
    | KaldiTransducer
    | NemoConformerCtc
    | NemoConformerRnnt
    | NemoConformerAED
    | TOneCtc
    | WhisperHf
    | WhisperOrt
)


def create_asr_resolver(
    model: str | None = None, local_dir: str | Path | None = None, *, offline: bool | None = None
) -> Resolver[AsrTypes]:
    """Create resolver for ASR models."""
    model_types: dict[str, type[AsrTypes]] = {
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
    return Resolver(model_types, model, local_dir, offline=offline)


VadTypes: TypeAlias = SileroVad | PyAnnoteVad


def create_vad_resolver(
    model: str | None = None, local_dir: str | Path | None = None, *, offline: bool | None = None
) -> Resolver[VadTypes]:
    """Create resolver for VAD models."""
    model_types: dict[str, type[VadTypes]] = {"silero": SileroVad, "pyannote": PyAnnoteVad}
    return Resolver(model_types, model, local_dir, offline=offline)


def create_se_resolver(
    model: str | None = None, local_dir: str | Path | None = None, *, offline: bool | None = None
) -> Resolver[WespeakerEmbeddings]:
    """Create resolver for SE models."""
    return Resolver(WespeakerEmbeddings, model, local_dir, offline=offline)


class PreprocessorRuntimeConfig(OnnxSessionOptions, total=False):
    """Preprocessor runtime config."""

    max_concurrent_workers: int | None
    """Max parallel preprocessing threads (None - auto, 1 - without parallel processing)."""

    use_numpy_preprocessors: bool | None
    """Use NumPy preprocessors backend instead of ONNX."""


class Manager:
    """Manager for models creation."""

    def __init__(
        self,
        sess_options: rt.SessionOptions | None = None,
        providers: Sequence[str | tuple[str, dict[Any, Any]]] | None = None,
        provider_options: Sequence[dict[Any, Any]] | None = None,
        preprocessor_config: PreprocessorRuntimeConfig | None = None,
        resampler_config: OnnxSessionOptions | None = None,
    ) -> None:
        """Create model manager."""
        self.default_onnx_config = update_onnx_providers(
            {"providers": rt.get_available_providers()}, excluded_providers=["AzureExecutionProvider"]
        ) | {
            "sess_options": sess_options,
            "providers": providers,
            "provider_options": provider_options,
        }

        if preprocessor_config is None:
            self.preprocessor_config = update_onnx_providers(
                self.default_onnx_config,
                new_options={"TensorrtExecutionProvider": {"trt_fp16_enable": False}},
                excluded_providers=OnnxPreprocessor._get_excluded_providers(),
            )
            self.preprocessor_max_workers: int | None = 1
            self.use_numpy_preprocessors = None
        else:
            self.preprocessor_max_workers = preprocessor_config.pop("max_concurrent_workers", 1)
            self.use_numpy_preprocessors = preprocessor_config.pop("use_numpy_preprocessors")
            self.preprocessor_config = preprocessor_config

        providers = get_onnx_providers(self.preprocessor_config)
        if self.use_numpy_preprocessors is None:
            self.use_numpy_preprocessors = not providers or providers == ["CPUExecutionProvider"]

        if resampler_config is None:
            resampler_config = update_onnx_providers(
                self.default_onnx_config, excluded_providers=Resampler._get_excluded_providers()
            )
        self.resampler_config = resampler_config

    def _create_preprocessor(self, name: str) -> Preprocessor:
        if name == "identity":
            return IdentityPreprocessor()

        preprocessor: Preprocessor
        if self.use_numpy_preprocessors:
            if name.startswith("gigaam"):
                preprocessor = GigaamPreprocessorNumpy(name)
            elif name in ("kaldi", "wespeaker"):
                preprocessor = KaldiPreprocessorNumpy(name)
            elif name.startswith("nemo"):
                preprocessor = NemoPreprocessorNumpy(name)
            elif name.startswith("whisper"):
                preprocessor = WhisperPreprocessorNumpy(name)
            else:
                raise ModelNotSupportedError(name)
        else:
            preprocessor = OnnxPreprocessor(name, self.preprocessor_config)

        if self.preprocessor_max_workers == 1:
            return preprocessor
        return ConcurrentPreprocessor(preprocessor, self.preprocessor_max_workers)

    def _create_resampler(self, sample_rate: Literal[8000, 16000]) -> Resampler:
        return Resampler(sample_rate, self.resampler_config)

    def _create_asr_adapter(self, asr: Asr) -> TextResultsAsrAdapter:
        return TextResultsAsrAdapter(asr, self._create_resampler(asr._get_sample_rate()))

    def _create_se_adapter(self, se: SpeakerEmbedding) -> SeAdapter:
        return SeAdapter(se, self._create_resampler(se._get_sample_rate()))

    def create_asr(
        self,
        model: str | ModelNames | ModelTypes | None = None,
        local_dir: str | Path | None = None,
        *,
        quantization: str | None = None,
        offline: bool | None = None,
        config: OnnxSessionOptions | None = None,
    ) -> TextResultsAsrAdapter:
        """Create ASR model."""
        resolver = create_asr_resolver(model, local_dir, offline=offline)
        if config is None:
            config = update_onnx_providers(
                self.default_onnx_config, excluded_providers=resolver.model_type._get_excluded_providers()
            )
        return self._create_asr_adapter(
            resolver.model_type(resolver.resolve_model(quantization=quantization), self._create_preprocessor, config)
        )

    def create_vad(
        self,
        model: str | VadNames | None = None,
        local_dir: str | Path | None = None,
        *,
        quantization: str | None = None,
        offline: bool | None = None,
        config: OnnxSessionOptions | None = None,
    ) -> Vad:
        """Create VAD model."""
        resolver = create_vad_resolver(model, local_dir, offline=offline)
        if config is None:
            config = update_onnx_providers(
                self.default_onnx_config, excluded_providers=resolver.model_type._get_excluded_providers()
            )
        return resolver.model_type(resolver.resolve_model(quantization=quantization), config)

    def create_se(
        self,
        model: str | None = None,
        local_dir: str | Path | None = None,
        *,
        quantization: str | None = None,
        offline: bool | None = None,
        config: OnnxSessionOptions | None = None,
    ) -> SeAdapter:
        """Create SE model."""
        resolver = create_se_resolver(model, local_dir, offline=offline)
        if config is None:
            config = update_onnx_providers(
                self.default_onnx_config, excluded_providers=resolver.model_type._get_excluded_providers()
            )
        return self._create_se_adapter(
            resolver.model_type(resolver.resolve_model(quantization=quantization), self._create_preprocessor, config)
        )


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

    manager = Manager(sess_options, providers, provider_options, preprocessor_config, resampler_config)
    return manager.create_asr(model, path, quantization=quantization, config=asr_config)


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
    manager = Manager()
    config: OnnxSessionOptions = {
        "sess_options": sess_options,
        "providers": providers,
        "provider_options": provider_options,
    }

    return manager.create_vad(
        model,
        path,
        quantization=quantization,
        config=config if any(value is not None for value in config.values()) else None,
    )
