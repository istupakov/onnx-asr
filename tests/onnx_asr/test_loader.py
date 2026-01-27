import sys
from pathlib import Path
from typing import get_args

import pytest

from onnx_asr.asr import Asr
from onnx_asr.loader import (
    AsrLoader,
    ModelNames,
    ModelTypes,
    VadLoader,
    VadNames,
)
from onnx_asr.models.kaldi import KaldiTransducer
from onnx_asr.models.nemo import NemoConformerAED
from onnx_asr.models.pyannote import PyAnnoteVad
from onnx_asr.models.silero import SileroVad
from onnx_asr.models.tone import TOneCtc
from onnx_asr.models.whisper import WhisperHf
from onnx_asr.utils import (
    InvalidModelTypeInConfigError,
    ModelFileNotFoundError,
    ModelNotSupportedError,
    ModelPathNotDirectoryError,
    MoreThanOneModelFileFoundError,
    NoModelNameOrPathSpecifiedError,
)
from onnx_asr.vad import Vad


@pytest.mark.parametrize("model", get_args(ModelNames))
def test_model_names(model: ModelNames) -> None:
    loader = AsrLoader(model)
    assert issubclass(loader._model_type, Asr)
    assert not loader.offline
    assert loader.local_dir is None
    assert isinstance(loader.repo_id, str)


@pytest.mark.parametrize("model", get_args(ModelNames))
def test_model_names_with_path(model: ModelNames, tmp_path: Path) -> None:
    loader = AsrLoader(model, tmp_path)
    assert issubclass(loader._model_type, Asr)
    assert loader.offline
    assert loader.local_dir == tmp_path
    assert isinstance(loader.repo_id, str)


@pytest.mark.parametrize(
    ("model", "type"),
    [
        ("alphacep/vosk-model-ru", KaldiTransducer),
        ("alphacep/vosk-model-small-ru", KaldiTransducer),
        ("t-tech/t-one", TOneCtc),
        ("onnx-community/whisper-tiny", WhisperHf),
        ("istupakov/canary-180m-flash-onnx", NemoConformerAED),
    ],
)
def test_model_repos(model: str, type: type[Asr]) -> None:
    loader = AsrLoader(model)
    assert loader._model_type == type
    assert not loader.offline
    assert loader.local_dir is None
    assert loader.repo_id == model


@pytest.mark.parametrize(
    ("model", "type"),
    [
        ("alphacep/vosk-model-ru", KaldiTransducer),
        ("alphacep/vosk-model-small-ru", KaldiTransducer),
        ("t-tech/t-one", TOneCtc),
    ],
)
def test_model_repos_with_path(model: str, tmp_path: Path, type: type[Asr]) -> None:
    loader = AsrLoader(model, tmp_path)
    assert loader._model_type == type
    assert loader.offline
    assert loader.local_dir == tmp_path
    assert loader.repo_id == model


@pytest.mark.parametrize("model", get_args(ModelTypes))
def test_model_types(model: ModelTypes, tmp_path: Path) -> None:
    loader = AsrLoader(model, tmp_path)
    assert issubclass(loader._model_type, Asr)
    assert loader.offline
    assert loader.local_dir == tmp_path
    assert loader.repo_id is None


def test_model_not_supported_error(tmp_path: Path) -> None:
    with pytest.raises(ModelNotSupportedError):
        AsrLoader("xxx", tmp_path)


@pytest.mark.parametrize("model", get_args(ModelTypes))
def test_no_model_name_or_path_specified_error(model: ModelTypes) -> None:
    with pytest.raises(NoModelNameOrPathSpecifiedError):
        AsrLoader(model)


@pytest.mark.parametrize("model", get_args(ModelTypes))
def test_no_model_name_and_empty_path_specified_error(model: ModelTypes, tmp_path: Path) -> None:
    with pytest.raises(NoModelNameOrPathSpecifiedError):
        AsrLoader(model, Path(tmp_path, "model"))


@pytest.mark.parametrize("model", get_args(ModelTypes))
def test_model_path_not_found_error(model: ModelTypes, tmp_path: Path) -> None:
    Path(tmp_path, "model").write_text("test")
    with pytest.raises(ModelPathNotDirectoryError):
        AsrLoader(model, Path(tmp_path, "model"))


def test_model_file_not_found_error(tmp_path: Path) -> None:
    with pytest.raises(ModelFileNotFoundError):
        AsrLoader("onnx-community/whisper-tiny", tmp_path)


def test_offline_model_file_not_found_error() -> None:
    with pytest.raises(ModelFileNotFoundError):
        AsrLoader("onnx-community/whisper-tiny", offline=True).resolve_model(quantization="fp16")


def test_invalid_model_type_in_config_error(tmp_path: Path) -> None:
    Path(tmp_path, "config.json").write_text('{"model_type": "xxx"}')
    with pytest.raises(InvalidModelTypeInConfigError):
        AsrLoader("onnx-community/whisper-tiny", tmp_path)


def test_remote_config_not_found_error() -> None:
    with pytest.raises(IOError):  # noqa: PT011
        AsrLoader("alphacep/vosk-model-small-ru").resolve_config()


def test_offline_config_not_found_error() -> None:
    with pytest.raises(FileNotFoundError):
        AsrLoader("alphacep/vosk-model-small-ru", offline=True).resolve_config()


def test_resolve_model_file_not_found_error() -> None:
    loader = AsrLoader("onnx-community/whisper-tiny")
    with pytest.raises(ModelFileNotFoundError):
        loader.resolve_model(quantization="xxx")


def test_more_than_one_model_file_found_error() -> None:
    loader = AsrLoader("onnx-community/whisper-tiny")
    with pytest.raises(MoreThanOneModelFileFoundError):
        loader.resolve_model(quantization="*int8")


def test_with_offline_huggingface_hub() -> None:
    AsrLoader("onnx-community/whisper-tiny").resolve_model(quantization="uint8")

    AsrLoader("onnx-community/whisper-tiny", offline=True).resolve_model(quantization="uint8")


def test_without_huggingface_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    loader = AsrLoader("onnx-community/whisper-tiny")

    path = loader._download_model("uint8", local_files_only=False)

    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    loader_with_path = AsrLoader("onnx-community/whisper-tiny", path)
    assert loader_with_path.offline
    loader_with_path.resolve_model(quantization="uint8")


@pytest.mark.parametrize("model", [*get_args(VadNames), "pyannote"])
def test_vad(model: str) -> None:
    loader = VadLoader(model)
    assert issubclass(loader._model_type, Vad)
    assert not loader.offline
    assert loader.local_dir is None
    assert isinstance(loader.repo_id, str)


@pytest.mark.parametrize("model", [*get_args(VadNames), "pyannote"])
def test_vad_with_path(model: str, tmp_path: Path) -> None:
    loader = VadLoader(model, tmp_path)
    assert issubclass(loader._model_type, SileroVad | PyAnnoteVad)
    assert loader.offline
    assert loader.local_dir == tmp_path
    assert isinstance(loader.repo_id, str)


def test_resolve_vad_file_not_found_error() -> None:
    loader = VadLoader("silero")
    with pytest.raises(ModelFileNotFoundError):
        loader.resolve_model(quantization="xxx")
