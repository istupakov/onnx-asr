"""Loader for ASR models."""

from pathlib import Path
from typing import Literal, get_args

from ._models import GigaamV2Ctc, GigaamV2Rnnt, KaldiTransducer, NemoConformerCtc, NemoConformerRnnt, Whisper
from .asr import Asr

ModelNames = Literal[
    "gigaam-v2-ctc",
    "gigaam-v2-rnnt",
    "nemo-fastconformer-ru-ctc",
    "nemo-fastconformer-ru-rnnt",
    "vosk-model-ru",
    "vosk-model-small-ru",
    "whisper-base",
]
ModelTypes = Literal["kaldi-rnnt", "nemo-conformer-ctc", "nemo-conformer-rnnt", "vosk", "whisper"]
ModelVersions = Literal["int8"] | None


def _get_model_class(name: ModelNames | ModelTypes):
    match name:
        case "gigaam-v2-ctc":
            return GigaamV2Ctc
        case "gigaam-v2-rnnt":
            return GigaamV2Rnnt
        case "kaldi-rnnt" | "vosk" | "vosk-model-ru" | "vosk-model-small-ru":
            return KaldiTransducer
        case "nemo-conformer-ctc" | "nemo-fastconformer-ru-ctc":
            return NemoConformerCtc
        case "nemo-conformer-rnnt" | "nemo-fastconformer-ru-rnnt":
            return NemoConformerRnnt
        case "whisper" | "whisper-base":
            return Whisper


def _resolve_paths(path: str | Path, model_files: dict[str, str]):
    assert Path(path).is_dir(), f"The path '{path}' is not a directory."

    def find(filename):
        files = list(Path(path).glob(filename))
        assert len(files) > 0, f"File '{filename}' not found in path '{path}'."
        assert len(files) == 1, f"Found more than 1 file '{filename}' found in path '{path}'."
        return files[0]

    return {key: find(filename) for key, filename in model_files.items()}


def _download_model(model: ModelNames, files: list[str]) -> str:
    from huggingface_hub import snapshot_download

    match model:
        case "gigaam-v2-ctc" | "gigaam-v2-rnnt":
            repo_id = "istupakov/gigaam-v2-onnx"
        case "nemo-fastconformer-ru-ctc" | "nemo-fastconformer-ru-rnnt":
            repo_id = "istupakov/stt_ru_fastconformer_hybrid_large_pc_onnx"
        case "vosk-model-ru" | "vosk-model-small-ru":
            repo_id = "alphacep/" + model
        case "whisper-base":
            repo_id = "istupakov/whisper-base-onnx"

    return snapshot_download(repo_id, allow_patterns=files)


def load_model(model: ModelNames | ModelTypes, path: str | Path | None = None, version: ModelVersions = None) -> Asr:
    """Load ASR model.

    Args:
        model: Model name or type:
                    GigaAM v2 (`gigaam-v2-ctc` | `gigaam-v2-rnnt`),
                    Kaldi Transducer (`kaldi-rnnt` | `vosk` | `vosk-model-ru` | `vosk-model-small-ru`)
                    Nvidia Conformer (`nemo-conformer-ctc` | `nemo-conformer-rnnt`)
                    Nvidia STT RU FastConformer Hybrid Large P&C (`nemo-fastconformer-ru-ctc` | `nemo-fastconformer-ru-rnnt`)
        path: Path to directory with model files.
        version: Model version: None for the default version or int8 for the quantized version.

    Returns:
        ASR model class.

    """
    model_class = _get_model_class(model)
    files = model_class._get_model_files(version)
    if path is None:
        assert model in get_args(ModelNames), "If the path is not specified, you must specify a specific model name."
        path = _download_model(model, list(files.values()))  # type: ignore

    return model_class(_resolve_paths(path, files))
