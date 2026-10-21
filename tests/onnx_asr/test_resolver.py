import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, get_args

import huggingface_hub
import pytest
from huggingface_hub.errors import LocalEntryNotFoundError
from huggingface_hub.utils._paths import filter_repo_objects

from onnx_asr.asr import Asr, BaseAsr
from onnx_asr.loader import (
    AsrNames,
    AsrTypeNames,
    Manager,
    VadNames,
    VadTypeNames,
    create_asr_resolver,
    create_se_resolver,
    create_vad_resolver,
    load_model,
    load_vad,
)
from onnx_asr.models.kaldi import KaldiTransducer
from onnx_asr.models.nemo import NemoConformerAED, NemoConformerTdt
from onnx_asr.models.tone import TOneCtc
from onnx_asr.models.wespeaker import WespeakerEmbeddings
from onnx_asr.models.whisper import WhisperHf
from onnx_asr.resolver import Resolver
from onnx_asr.utils import (
    InvalidModelTypeInConfigError,
    ModelFileNotFoundError,
    ModelNotSupportedError,
    ModelPathNotDirectoryError,
    MoreThanOneModelFileFoundError,
    NoModelNameOrPathSpecifiedError,
)
from onnx_asr.vad import BaseVad

NEMO_FILES = {
    "encoder": "encoder-model.int8.onnx",
    "decoder_joint": "decoder_joint-model.int8.onnx",
    "vocab": "vocab.txt",
}


def write_files(root: Path, files: dict[str, str]) -> dict[str, Path]:
    paths = {}
    for key, filename in files.items():
        path = root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test")
        paths[key] = path
    return paths


def mock_snapshot_download(root: Path, monkeypatch: pytest.MonkeyPatch, files: dict[str, str]) -> None:
    def download(_repo_id: str, **kwargs: Any) -> str:
        if kwargs["local_files_only"]:
            msg = "cache miss"
            raise LocalEntryNotFoundError(msg)
        selected = set(filter_repo_objects(files.values(), allow_patterns=kwargs["allow_patterns"]))
        write_files(root, {key: file for key, file in files.items() if file in selected})
        return str(root)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)


# Model selection.


@pytest.mark.parametrize("model", get_args(AsrNames))
def test_model_names(model: AsrNames) -> None:
    resolver = create_asr_resolver(model)
    assert issubclass(resolver.model_type, BaseAsr)
    assert not resolver.offline
    assert resolver.local_dir is None
    assert isinstance(resolver.repo_id, str)


@pytest.mark.parametrize("model", get_args(AsrNames))
def test_model_names_with_path(model: AsrNames, tmp_path: Path) -> None:
    resolver = create_asr_resolver(model, tmp_path)
    assert issubclass(resolver.model_type, BaseAsr)
    assert resolver.offline
    assert resolver.local_dir == tmp_path
    assert isinstance(resolver.repo_id, str)


@pytest.mark.parametrize(
    ("model", "expected_type"),
    [
        ("alphacep/vosk-model-ru", KaldiTransducer),
        ("alphacep/vosk-model-small-ru", KaldiTransducer),
        ("t-tech/t-one", TOneCtc),
    ],
)
def test_model_repos_with_path(model: str, tmp_path: Path, expected_type: type[Asr]) -> None:
    resolver = create_asr_resolver(model, tmp_path)
    assert resolver.model_type == expected_type
    assert resolver.offline
    assert resolver.local_dir == tmp_path
    assert resolver.repo_id == model


@pytest.mark.parametrize("model", get_args(AsrTypeNames))
def test_model_types(model: AsrTypeNames, tmp_path: Path) -> None:
    resolver = create_asr_resolver(model, tmp_path)
    assert issubclass(resolver.model_type, BaseAsr)
    assert resolver.offline
    assert resolver.local_dir == tmp_path
    assert resolver.repo_id is None


def test_custom_model_type() -> None:
    resolver = Resolver(KaldiTransducer, "alphacep/vosk-model-ru")
    assert resolver.model_type == KaldiTransducer
    assert not resolver.offline
    assert resolver.local_dir is None
    assert resolver.repo_id == "alphacep/vosk-model-ru"


def test_custom_model_type_with_path(tmp_path: Path) -> None:
    resolver = Resolver(KaldiTransducer, local_dir=tmp_path)
    assert resolver.model_type == KaldiTransducer
    assert resolver.offline
    assert resolver.local_dir == tmp_path
    assert resolver.repo_id is None


def test_vad_with_path(tmp_path: Path) -> None:
    resolver = create_vad_resolver("silero", tmp_path)
    assert issubclass(resolver.model_type, BaseVad)
    assert resolver.offline
    assert resolver.local_dir == tmp_path
    assert isinstance(resolver.repo_id, str)


@pytest.mark.parametrize("model", get_args(VadTypeNames))
def test_vad_types(model: VadTypeNames, tmp_path: Path) -> None:
    resolver = create_vad_resolver(model, tmp_path)
    assert issubclass(resolver.model_type, BaseVad)
    assert resolver.offline
    assert resolver.local_dir == tmp_path
    assert resolver.repo_id is None


def test_se() -> None:
    resolver = create_se_resolver("wespeaker/wespeaker-voxceleb-resnet34")
    assert issubclass(resolver.model_type, WespeakerEmbeddings)
    assert not resolver.offline
    assert resolver.local_dir is None
    assert isinstance(resolver.repo_id, str)


def test_se_with_path(tmp_path: Path) -> None:
    resolver = create_se_resolver(local_dir=tmp_path)
    assert issubclass(resolver.model_type, WespeakerEmbeddings)
    assert resolver.offline
    assert resolver.local_dir == tmp_path
    assert resolver.repo_id is None


# Argument and configuration validation.


def test_model_not_supported_error(tmp_path: Path) -> None:
    with pytest.raises(ModelNotSupportedError):
        create_asr_resolver("xxx", tmp_path)


@pytest.mark.parametrize("model", get_args(AsrTypeNames))
def test_no_model_name_or_path_specified_error(model: AsrTypeNames) -> None:
    with pytest.raises(NoModelNameOrPathSpecifiedError):
        create_asr_resolver(model)


@pytest.mark.parametrize("model", get_args(AsrTypeNames))
def test_missing_model_and_nonexistent_local_dir_error(model: AsrTypeNames, tmp_path: Path) -> None:
    with pytest.raises(NoModelNameOrPathSpecifiedError):
        create_asr_resolver(model, Path(tmp_path, "model"))


@pytest.mark.parametrize("model", get_args(AsrTypeNames))
def test_local_dir_is_file_error(model: AsrTypeNames, tmp_path: Path) -> None:
    Path(tmp_path, "model").write_text("test")
    with pytest.raises(ModelPathNotDirectoryError):
        create_asr_resolver(model, Path(tmp_path, "model"))


@pytest.mark.parametrize("subfolder", ["..", "../export", "export/../model", "/export"])
def test_invalid_subfolder(tmp_path: Path, subfolder: str) -> None:
    with pytest.raises(ValueError, match="Subfolder must be a relative path"):
        create_asr_resolver("nemo-conformer-tdt", tmp_path, subfolder=subfolder)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows path semantics")
@pytest.mark.parametrize("subfolder", [r"\export", "C:export", "C:/export", r"\\server\share\export"])
def test_invalid_windows_subfolder(tmp_path: Path, subfolder: str) -> None:
    with pytest.raises(ValueError, match="Subfolder must be a relative path"):
        create_asr_resolver("nemo-conformer-tdt", tmp_path, subfolder=subfolder)


@pytest.mark.parametrize("subfolder", ["**/exports/**", "exports/**/variants/**/int8"])
def test_subfolder_rejects_multiple_recursive_components(tmp_path: Path, subfolder: str) -> None:
    with pytest.raises(ValueError, match="At most one"):
        create_asr_resolver("nemo-conformer-tdt", tmp_path, subfolder=subfolder)


@pytest.mark.parametrize(
    "config_json",
    [
        '{"model_type": "xxx"}',
        '{"model_type": "alphacep/vosk-model-ru"}',
        "{}",
        '{"model_type": null}',
        '{"model_type": 1}',
        '{"model_type": false}',
        '{"model_type": []}',
        '{"model_type": {}}',
        "[]",
        "null",
        '"text"',
        "1",
        "false",
    ],
)
def test_invalid_model_type_in_config_error(tmp_path: Path, config_json: str) -> None:
    Path(tmp_path, "config.json").write_text(config_json)
    with pytest.raises(InvalidModelTypeInConfigError):
        create_asr_resolver("onnx-community/whisper-tiny", tmp_path)


# Local model resolution.


@pytest.mark.parametrize(
    ("subfolder", "directory"),
    [
        (None, ""),
        ("", ""),
        ("onnx/combined-v0.1.0-int8", "onnx/combined-v0.1.0-int8"),
        ("*", "export"),
        ("**", "nested/export"),
        ("exports/**/int8/*", "exports/int8/model"),
        ("exports/**/int8/*", "exports/nested/int8/model"),
    ],
)
def test_config_selects_complete_export(tmp_path: Path, subfolder: str | None, directory: str) -> None:
    model_dir = tmp_path / directory
    expected = write_files(model_dir, NEMO_FILES)
    config_path = model_dir / "config.json"
    config_path.write_text('{"model_type": "nemo-conformer-tdt"}')
    write_files(tmp_path / "other", NEMO_FILES)
    resolver = create_asr_resolver("example/model", tmp_path, subfolder=subfolder, revision="release")
    assert resolver.offline
    assert resolver.model_type is NemoConformerTdt
    assert resolver.resolve_config() == config_path
    assert resolver.resolve_model(quantization="int8") == expected | {"config": config_path}


@pytest.mark.parametrize(
    ("model", "graph_dir"),
    [
        ("alphacep/vosk-model-ru", "am-onnx"),
        ("alphacep/vosk-model-small-ru", "am"),
        ("kaldi-rnnt", "am"),
    ],
)
@pytest.mark.parametrize("subfolder", ["", "export"])
def test_kaldi_file_layout(tmp_path: Path, model: str, graph_dir: str, subfolder: str) -> None:
    expected = write_files(
        tmp_path / subfolder,
        {
            "encoder": f"{graph_dir}/encoder.int8.onnx",
            "decoder": f"{graph_dir}/decoder.int8.onnx",
            "joiner": f"{graph_dir}/joiner.int8.onnx",
            "vocab": "lang/tokens.txt",
        },
    )
    resolver = create_asr_resolver(model, tmp_path, subfolder=subfolder)
    assert resolver.resolve_model(quantization="int8") == expected


@pytest.mark.parametrize("subfolder", ["", "export"])
def test_whisper_file_layout(tmp_path: Path, subfolder: str) -> None:
    expected = write_files(
        tmp_path / subfolder,
        {
            "encoder": "onnx/encoder_model.int8.onnx",
            "decoder": "onnx/decoder_model_merged.int8.onnx",
            "vocab": "vocab.json",
            "added_tokens": "added_tokens.json",
            "config": "config.json",
        },
    )
    expected["config"].write_text('{"model_type": "whisper"}')
    resolver = create_asr_resolver("onnx-community/whisper-tiny", tmp_path, subfolder=subfolder)
    assert resolver.resolve_model(quantization="int8") == expected


@pytest.mark.parametrize("subfolder", ["", "export"])
def test_pyannote_file_layout(tmp_path: Path, subfolder: str) -> None:
    expected = write_files(tmp_path / subfolder, {"model": "onnx/model.int8.onnx", "config": "config.json"})
    expected["config"].write_text('{"model_type": "pyannote"}')
    resolver = create_vad_resolver("onnx-community/pyannote-segmentation-3.0", tmp_path, subfolder=subfolder)
    assert resolver.resolve_model(quantization="int8") == expected


@pytest.mark.parametrize("subfolder", ["", "export"])
def test_custom_model_without_config(tmp_path: Path, subfolder: str) -> None:
    class CustomModel:
        @staticmethod
        def _get_model_files(quantization: str | None = None) -> dict[str, str]:
            return {"model": "model.onnx"}

    expected = write_files(tmp_path / subfolder, {"model": "model.onnx"})
    resolver = Resolver(CustomModel, local_dir=tmp_path, subfolder=subfolder)
    assert resolver.resolve_model() == expected


def test_without_huggingface_hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    expected = write_files(tmp_path, NEMO_FILES)
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    resolver = create_asr_resolver("nemo-conformer-tdt", tmp_path)
    assert resolver.offline
    assert resolver.resolve_model(quantization="int8") == expected


def test_missing_config_for_model_type_detection(tmp_path: Path) -> None:
    with pytest.raises(ModelFileNotFoundError):
        create_asr_resolver("onnx-community/whisper-tiny", tmp_path)


@pytest.mark.parametrize("subfolder", ["*", "**"])
def test_glob_requires_config(tmp_path: Path, subfolder: str) -> None:
    write_files(tmp_path / "first", NEMO_FILES)
    resolver = create_asr_resolver("nemo-conformer-tdt", tmp_path, subfolder=subfolder)
    with pytest.raises(ModelFileNotFoundError):
        resolver.resolve_model(quantization="int8")


@pytest.mark.parametrize("subfolder", ["*", "**"])
def test_glob_requires_unique_config(tmp_path: Path, subfolder: str) -> None:
    for directory in ("first", "second"):
        write_files(tmp_path / directory, {"config": "config.json"})
    resolver = create_asr_resolver("nemo-conformer-tdt", tmp_path, subfolder=subfolder)
    with pytest.raises(MoreThanOneModelFileFoundError):
        resolver.resolve_model(quantization="int8")


def test_missing_vocabulary_does_not_fall_back_to_root(tmp_path: Path) -> None:
    expected = write_files(tmp_path / "export", NEMO_FILES)
    expected["vocab"].rename(tmp_path / "vocab.txt")
    resolver = create_asr_resolver("nemo-conformer-tdt", tmp_path, subfolder="export")
    with pytest.raises(ModelFileNotFoundError):
        resolver.resolve_model(quantization="int8")


# Model file search.


@pytest.mark.parametrize("suffix", ["onnx", "ort"])
def test_model_file_pattern_ambiguity(tmp_path: Path, suffix: str) -> None:
    for directory in ("first", "second"):
        write_files(
            tmp_path / directory,
            {"encoder": f"onnx/encoder_model.int8.{suffix}", "decoder": f"onnx/decoder_model_merged.int8.{suffix}"},
        )
    resolver = create_asr_resolver("whisper", tmp_path)
    with pytest.raises(MoreThanOneModelFileFoundError):
        resolver.resolve_model(quantization="int8")


@pytest.mark.parametrize(
    ("pattern", "filename"),
    [("**/config.json", "config.json"), ("**/model.onnx", "model.onnx"), ("**/model.onnx", "model.ort")],
)
def test_file_search_ignores_directories(tmp_path: Path, pattern: str, filename: str) -> None:
    for name in {Path(pattern).name, filename}:
        (tmp_path / "decoy" / name).mkdir(parents=True, exist_ok=True)
    expected = write_files(tmp_path / "export", {"file": filename})
    assert Resolver._find_file(tmp_path, pattern) == expected["file"]


def test_onnx_preferred_over_ort(tmp_path: Path) -> None:
    expected = write_files(tmp_path, NEMO_FILES)
    write_files(tmp_path, {"encoder": "encoder-model.int8.ort", "decoder_joint": "decoder_joint-model.int8.ort"})
    resolver = create_asr_resolver("nemo-conformer-tdt", tmp_path)
    assert resolver.resolve_model(quantization="int8") == expected


def test_subfolder_ort_fallback(tmp_path: Path) -> None:
    expected = write_files(tmp_path / "export", NEMO_FILES)
    for key in ("encoder", "decoder_joint"):
        path = expected[key]
        path.rename(path.with_suffix(".ort"))
        expected[key] = path.with_suffix(".ort")
    resolver = create_asr_resolver("nemo-conformer-tdt", tmp_path, subfolder="export")
    assert resolver.resolve_model(quantization="int8") == expected


# Downloads without network access.


@pytest.mark.parametrize(("cached", "offline"), [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("subfolder", ["onnx/release", "**"])
def test_remote_revision_and_subfolder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cached: bool, offline: bool, subfolder: str
) -> None:
    model_subfolder = "onnx/release"
    expected = write_files(tmp_path / model_subfolder, NEMO_FILES)
    config_path = tmp_path / model_subfolder / "config.json"
    config_path.write_text('{"model_type": "nemo-conformer-tdt"}')
    calls: list[dict[str, Any]] = []

    def download(repo_id: str, **kwargs: Any) -> str:
        calls.append(kwargs)
        assert repo_id == "example/model"
        assert kwargs["revision"] == "fixed-commit"
        if kwargs["local_files_only"] and not cached:
            msg = "cache miss"
            raise LocalEntryNotFoundError(msg)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    resolver = create_asr_resolver("example/model", subfolder=subfolder, revision="fixed-commit", offline=offline)
    assert resolver.resolve_model(quantization="int8") == expected | {"config": config_path}
    flags = [True] if cached else [True, False]
    assert [call["local_files_only"] for call in calls] == flags * 2
    config_patterns = ["**/config.json", "config.json"] if subfolder == "**" else [f"{subfolder}/config.json"]
    assert calls[0]["allow_patterns"] == config_patterns
    assert set(calls[-1]["allow_patterns"]) == {
        f"{model_subfolder}/config.json",
        f"{model_subfolder}/config.yaml",
        f"{model_subfolder}/vocab.txt",
        f"{model_subfolder}/encoder-model?int8.onnx",
        f"{model_subfolder}/decoder_joint-model?int8.onnx",
        f"{model_subfolder}/encoder-model?int8.ort",
        f"{model_subfolder}/decoder_joint-model?int8.ort",
        f"{model_subfolder}/encoder-model?int8.onnx?data",
        f"{model_subfolder}/decoder_joint-model?int8.onnx?data",
    }


def test_glob_download_uses_selected_config_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_dir = tmp_path / "exports/selected"
    expected = write_files(model_dir, NEMO_FILES)
    config_path = model_dir / "config.json"
    config_path.write_text('{"model_type": "nemo-conformer-tdt"}')
    write_files(tmp_path / "other", NEMO_FILES)
    calls: list[dict[str, Any]] = []

    def download(_repo_id: str, **kwargs: Any) -> str:
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    resolver = create_asr_resolver("nemo-parakeet-tdt-0.6b-v3", subfolder="**", revision="release", offline=True)
    assert resolver.resolve_model(quantization="int8") == expected | {"config": config_path}
    assert calls[0]["allow_patterns"] == ["**/config.json", "config.json"]
    assert all(pattern.startswith("exports/selected/") for pattern in calls[1]["allow_patterns"])
    assert "exports/selected/encoder-model?int8.onnx?data" in calls[1]["allow_patterns"]
    assert all(call["local_files_only"] and call["revision"] == "release" for call in calls)
    assert len(calls) == 2
    assert resolver.resolve_model(quantization="int8") == expected | {"config": config_path}
    assert len(calls) == 3
    assert calls[2]["allow_patterns"] == calls[1]["allow_patterns"]


@pytest.mark.parametrize("directory", ["", "nested/deep"])
@pytest.mark.parametrize("graph_dir", ["", "onnx/deep"])
def test_recursive_download_patterns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, directory: str, graph_dir: str
) -> None:
    class CustomModel:
        @staticmethod
        def _get_model_files(quantization: str | None = None) -> dict[str, str]:
            return {"model": "**/model.onnx"}

    model_dir = Path("exports") / directory / "int8"
    repo_files = {
        "config": (model_dir / "config.json").as_posix(),
        "model": (model_dir / graph_dir / "model.onnx").as_posix(),
    }

    mock_snapshot_download(tmp_path, monkeypatch, repo_files)
    resolver = Resolver(CustomModel, "example/model", subfolder="exports/**/int8")
    assert resolver.resolve_model() == {key: tmp_path / file for key, file in repo_files.items()}


def test_download_ort_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_files = {
        "encoder": "export/encoder-model.int8.ort",
        "decoder_joint": "export/decoder_joint-model.int8.ort",
        "vocab": "export/vocab.txt",
    }

    mock_snapshot_download(tmp_path, monkeypatch, repo_files)
    resolver = create_asr_resolver("nemo-parakeet-tdt-0.6b-v3", subfolder="export")
    assert resolver.resolve_model(quantization="int8") == {key: tmp_path / file for key, file in repo_files.items()}


@pytest.mark.parametrize("data_suffix", [".data", "_data"])
def test_download_external_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, data_suffix: str) -> None:
    repo_files = {key: f"export/{file}" for key, file in NEMO_FILES.items()}
    data_file = repo_files["encoder"] + data_suffix
    repo_files["data"] = data_file

    mock_snapshot_download(tmp_path, monkeypatch, repo_files)
    resolver = create_asr_resolver("nemo-parakeet-tdt-0.6b-v3", subfolder="export")
    assert resolver.resolve_model(quantization="int8") == {
        key: tmp_path / file for key, file in repo_files.items() if key != "data"
    }
    assert (tmp_path / data_file).is_file()


def test_download_rejects_multiple_recursive_components(monkeypatch: pytest.MonkeyPatch) -> None:
    class CustomModel:
        @staticmethod
        def _get_model_files(quantization: str | None = None) -> dict[str, str]:
            return {"model": "**/onnx/**/model.onnx"}

    def download(*_args: object, **_kwargs: Any) -> str:
        pytest.fail("Invalid patterns must be rejected before downloading")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    resolver = Resolver(CustomModel, "example/model", offline=True)
    with pytest.raises(ValueError, match="At most one"):
        resolver.resolve_model()


@pytest.mark.parametrize("offline", [False, True])
def test_missing_config_does_not_fall_back_to_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, offline: bool
) -> None:
    (tmp_path / "config.json").write_text('{"model_type": "nemo-conformer-tdt"}')
    calls: list[dict[str, Any]] = []

    def download(_repo_id: str, **kwargs: Any) -> str:
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    with pytest.raises(ModelFileNotFoundError):
        create_asr_resolver("example/model", subfolder="export", offline=offline)
    assert all(call["allow_patterns"] == ["export/config.json"] for call in calls)
    if offline:
        assert all(call["local_files_only"] for call in calls)


def test_offline_incomplete_snapshot_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_incomplete(*_args: object, **_kwargs: object) -> str:
        message = "incomplete snapshot"
        raise LocalEntryNotFoundError(message)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", raise_incomplete)
    with pytest.raises(ModelFileNotFoundError):
        create_asr_resolver("onnx-community/whisper-tiny", offline=True).resolve_model(quantization="fp16")


# Public loading functions.


@pytest.mark.parametrize("load_function", [load_model, load_vad, Manager.create_se])
def test_public_parameters_reach_resolver(
    load_function: Callable[..., object], monkeypatch: pytest.MonkeyPatch
) -> None:
    def create_resolver(*_args: object, **kwargs: Any) -> None:
        assert kwargs["subfolder"] == "export"
        assert kwargs["revision"] == "release"
        msg = "resolver reached"
        raise RuntimeError(msg)

    monkeypatch.setattr("onnx_asr.loader.Resolver", create_resolver)
    if load_function is Manager.create_se:
        load_function = Manager().create_se
    with pytest.raises(RuntimeError, match="resolver reached"):
        load_function("example/model", subfolder="export", revision="release")


# Integration tests using Hugging Face repositories or cache.


@pytest.mark.parametrize(
    ("model", "expected_type"),
    [
        ("alphacep/vosk-model-ru", KaldiTransducer),
        ("alphacep/vosk-model-small-ru", KaldiTransducer),
        ("t-tech/t-one", TOneCtc),
        ("onnx-community/whisper-tiny", WhisperHf),
        ("istupakov/canary-180m-flash-onnx", NemoConformerAED),
    ],
)
def test_model_repos(model: str, expected_type: type[Asr]) -> None:
    resolver = create_asr_resolver(model)
    assert resolver.model_type == expected_type
    assert not resolver.offline
    assert resolver.local_dir is None
    assert resolver.repo_id == model


@pytest.mark.parametrize("model", get_args(VadNames))
def test_vad(model: str) -> None:
    resolver = create_vad_resolver(model)
    assert issubclass(resolver.model_type, BaseVad)
    assert not resolver.offline
    assert resolver.local_dir is None
    assert isinstance(resolver.repo_id, str)


def test_offline_model_file_not_found_error() -> None:
    with pytest.raises(ModelFileNotFoundError):
        create_asr_resolver("onnx-community/whisper-tiny", offline=True).resolve_model(quantization="fp16")


def test_remote_config_not_found_error() -> None:
    with pytest.raises(IOError):  # noqa: PT011
        create_asr_resolver("alphacep/vosk-model-small-ru").resolve_config()


def test_offline_config_not_found_error() -> None:
    with pytest.raises(FileNotFoundError):
        create_asr_resolver("alphacep/vosk-model-small-ru", offline=True).resolve_config()


def test_resolve_model_file_not_found_error() -> None:
    resolver = create_asr_resolver("onnx-community/whisper-tiny")
    with pytest.raises(ModelFileNotFoundError):
        resolver.resolve_model(quantization="xxx")


def test_resolve_vad_file_not_found_error() -> None:
    resolver = create_vad_resolver("silero")
    with pytest.raises(ModelFileNotFoundError):
        resolver.resolve_model(quantization="xxx")


def test_more_than_one_model_file_found_error() -> None:
    resolver = create_asr_resolver("onnx-community/whisper-tiny.en")
    with pytest.raises(MoreThanOneModelFileFoundError):
        resolver.resolve_model(quantization="*int8")


def test_offline_loading_after_download() -> None:
    create_asr_resolver("onnx-community/whisper-tiny").resolve_model(quantization="uint8")

    create_asr_resolver("onnx-community/whisper-tiny", offline=True).resolve_model(quantization="uint8")
