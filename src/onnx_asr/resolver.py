"""Resolve model files from local directories or Hugging Face Hub."""

import json
from glob import has_magic
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
    "gigaam-multilingual-ctc": "istupakov/gigaam-multilingual-ctc-onnx",
    "gigaam-multilingual-large-ctc": "istupakov/gigaam-multilingual-large-ctc-onnx",
    "nemo-fastconformer-ru-ctc": "istupakov/stt_ru_fastconformer_hybrid_large_pc_onnx",
    "nemo-fastconformer-ru-rnnt": "istupakov/stt_ru_fastconformer_hybrid_large_pc_onnx",
    "nemo-parakeet-ctc-0.6b": "istupakov/parakeet-ctc-0.6b-onnx",
    "nemo-parakeet-rnnt-0.6b": "istupakov/parakeet-rnnt-0.6b-onnx",
    "nemo-parakeet-tdt-0.6b-v2": "istupakov/parakeet-tdt-0.6b-v2-onnx",
    "nemo-parakeet-tdt-0.6b-v3": "istupakov/parakeet-tdt-0.6b-v3-onnx",
    "nemo-canary-1b-v2": "istupakov/canary-1b-v2-onnx",
    "whisper-base": "istupakov/whisper-base-onnx",
    "silero": "istupakov/silero-vad-onnx",
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
    model_type: type[T]
    revision: str | None
    subfolder: Path | None

    def __init__(  # noqa: C901
        self,
        model_type: type[T] | dict[str, type[T]],
        model: str | None = None,
        local_dir: str | Path | None = None,
        *,
        subfolder: str | None = None,
        revision: str | None = None,
        offline: bool | None = None,
    ):
        """Create model resolver."""
        subfolder_pattern = Path(subfolder or "")
        if subfolder_pattern.anchor or ".." in subfolder_pattern.parts:
            msg = f"Subfolder must be a relative path without '..': {subfolder_pattern}"
            raise ValueError(msg)
        if subfolder_pattern.parts.count("**") > 1:
            msg = f"At most one '**' component is allowed in subfolder: {subfolder_pattern}"
            raise ValueError(msg)
        self.revision = revision
        self._config_pattern = (subfolder_pattern / "config.json").as_posix()
        self.subfolder = None if has_magic(subfolder_pattern.as_posix()) else subfolder_pattern

        if model is not None:
            if "/" in model:
                self.repo_id = model
            elif model in model_repos:
                self.repo_id = model_repos[model]
            elif not (isinstance(model_type, dict) and model in model_type):
                raise ModelNotSupportedError(model)

        if local_dir is not None:
            self.local_dir = Path(local_dir)
            if self.local_dir.exists() and not self.local_dir.is_dir():
                raise ModelPathNotDirectoryError(self.local_dir)

        self.offline = offline if offline is not None else self.local_dir is not None and self.local_dir.exists()

        if not (self.repo_id or (self.offline and self.local_dir)):
            raise NoModelNameOrPathSpecifiedError

        if isinstance(model_type, type):
            self.model_type = model_type
        elif model in model_type:
            self.model_type = model_type[model]
        else:
            with self.resolve_config().open("rt", encoding="utf-8") as f:
                config = json.load(f)

            config_model_type = config.get("model_type") if isinstance(config, dict) else None
            if (
                not isinstance(config_model_type, str)
                or "/" in config_model_type
                or config_model_type not in model_type
            ):
                raise InvalidModelTypeInConfigError(str(config_model_type))
            self.model_type = model_type[config_model_type]

    def _resolve_model_path(self, snapshot_path: Path) -> Path:
        if self.subfolder is None:
            config_path = self._find_file(snapshot_path, self._config_pattern)
            self.subfolder = config_path.parent.relative_to(snapshot_path)
        return snapshot_path / self.subfolder

    def _download_config(self, *, local_files_only: bool) -> Path:
        snapshot_path = self._download([self._config_pattern], local_files_only=local_files_only)
        return self._find_file(self._resolve_model_path(snapshot_path), "config.json")

    def _download_model(self, quantization: str | None, *, local_files_only: bool) -> Path:
        if self.subfolder is None:
            self._download_config(local_files_only=local_files_only)
        assert self.subfolder is not None
        files = ["config.json", "config.yaml", *self.model_type._get_model_files(quantization).values()]
        files += [
            str(file_path.with_suffix(suffix))
            for file in files
            if (file_path := Path(file)).suffix == ".onnx"
            for suffix in (".ort", ".onnx?data")
        ]
        snapshot_path = self._download(
            [(self.subfolder / file).as_posix() for file in files], local_files_only=local_files_only
        )
        return snapshot_path / self.subfolder

    def _download(self, files: list[str], *, local_files_only: bool) -> Path:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
        from huggingface_hub.errors import LocalEntryNotFoundError  # noqa: PLC0415

        for file in files:
            if Path(file).parts.count("**") > 1:
                msg = f"At most one '**' component is allowed in file pattern: {file}"
                raise ValueError(msg)
        # Hub filters require a separate pattern for zero directory levels.
        files = [
            *files,
            *(file.replace("**/", "") for file in files if "**/" in file),
        ]

        assert self.repo_id is not None
        try:
            return Path(
                snapshot_download(
                    self.repo_id,
                    revision=self.revision,
                    local_dir=self.local_dir,
                    local_files_only=local_files_only,
                    allow_patterns=files,
                )
            )
        except LocalEntryNotFoundError as e:
            # Use the same missing-file error for local paths and Hub snapshots.
            filename = "<missing files in local snapshot>"
            raise ModelFileNotFoundError(filename, self.repo_id) from e

    @staticmethod
    def _find_file(path: Path, filename: str) -> Path:
        files = [file for file in path.glob(filename) if file.is_file()]
        if not files and Path(filename).suffix == ".onnx":
            files = [file for file in path.glob(str(Path(filename).with_suffix(".ort"))) if file.is_file()]
        if len(files) > 1:
            raise MoreThanOneModelFileFoundError(filename, path)
        if files:
            return files[0]
        raise ModelFileNotFoundError(filename, path)

    def _resolve_model_files(self, path: Path, quantization: str | None) -> dict[str, Path]:
        files = {
            key: self._find_file(path, filename)
            for key, filename in self.model_type._get_model_files(quantization).items()
        }
        config_path = path / "config.json"
        if config_path.is_file():
            files["config"] = config_path
        return files

    def resolve_config(self) -> Path:
        """Resolve path to model config."""
        try:
            if self.local_dir:
                return self._find_file(self._resolve_model_path(self.local_dir), "config.json")
            return self._download_config(local_files_only=True)
        except FileNotFoundError:
            if self.offline:
                raise
            return self._download_config(local_files_only=False)

    def resolve_model(self, *, quantization: str | None = None) -> dict[str, Path]:
        """Resolve paths to model files."""
        try:
            model_path = (
                self._resolve_model_path(self.local_dir)
                if self.local_dir
                else self._download_model(quantization, local_files_only=True)
            )
            return self._resolve_model_files(model_path, quantization)
        except FileNotFoundError:
            if self.offline:
                raise
            return self._resolve_model_files(self._download_model(quantization, local_files_only=False), quantization)
