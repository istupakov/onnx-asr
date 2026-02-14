"""Resolver for ASR and VAD models."""

import json
from pathlib import Path
from typing import Generic, Protocol, TypeVar

from onnx_asr.utils import (
    InvalidModelTypeInConfigError,
    ModelFileNotFoundError,
    ModelNotSupportedError,
    ModelPathNotDirectoryError,
    MoreThanOneModelFileFoundError,
    NoModelNameOrPathSpecifiedError,
)

model_repos = {
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
    "silero": "onnx-community/silero-vad",
    "pyannote": "onnx-community/pyannote-segmentation-3.0",
}


class _Model(Protocol):
    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]: ...


T = TypeVar("T", bound=_Model)


class Resolver(Generic[T]):
    """Model resolver."""

    offline: bool = False
    local_dir: Path | None = None
    repo_id: str | None = None

    def __init__(  # noqa: C901
        self,
        model_type: type[T] | dict[str, type[T]],
        model: str | None = None,
        local_dir: str | Path | None = None,
        *,
        offline: bool | None = None,
    ):
        """Create model loader."""
        if local_dir is not None:
            self.local_dir = Path(local_dir)
            if self.local_dir.exists():
                self.offline = True
                if not self.local_dir.is_dir():
                    raise ModelPathNotDirectoryError(self.local_dir)

        if offline is not None:
            self.offline = offline

        if model and "/" in model:
            self.repo_id = model
        elif model in model_repos:
            self.repo_id = model_repos[model]
        elif not (self.offline and self.local_dir):
            raise NoModelNameOrPathSpecifiedError

        if isinstance(model_type, type):
            self.model_type = model_type
        elif model in model_type:
            self.model_type = model_type[model]
        elif not model or "/" in model:
            with self.resolve_config().open("rt", encoding="utf-8") as f:
                config = json.load(f)

            config_model_type: str = config.get("model_type")
            if "/" in config_model_type or config_model_type not in model_type:
                raise InvalidModelTypeInConfigError(config_model_type)
            self.model_type = model_type[config_model_type]
        else:
            raise ModelNotSupportedError(model)

    def _download_config(self, *, local_files_only: bool) -> Path:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        assert self.repo_id is not None
        return Path(
            hf_hub_download(self.repo_id, "config.json", local_dir=self.local_dir, local_files_only=local_files_only)  # nosec
        )

    def _download_model(self, quantization: str | None, *, local_files_only: bool) -> Path:
        from huggingface_hub import snapshot_download  # noqa: PLC0415

        files = list(self.model_type._get_model_files(quantization).values())
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
        files = self.model_type._get_model_files(quantization)
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
        """Resolve path to model config."""
        try:
            if self.local_dir:
                config_path = Path(self.local_dir, "config.json")
                if not config_path.is_file():
                    raise ModelFileNotFoundError(config_path.name, self.local_dir)
                return config_path

            return self._download_config(local_files_only=True)
        except FileNotFoundError:
            if self.offline:
                raise
            return self._download_config(local_files_only=False)

    def resolve_model(self, *, quantization: str | None = None) -> dict[str, Path]:
        """Resolve paths to model files."""
        try:
            return self._resolve_model_files(
                self.local_dir or self._download_model(quantization, local_files_only=True), quantization
            )
        except FileNotFoundError:
            if self.offline:
                raise
            return self._resolve_model_files(self._download_model(quantization, local_files_only=False), quantization)
