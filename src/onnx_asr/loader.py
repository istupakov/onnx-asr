"""Loader for ASR models."""

from pathlib import Path
from typing import Literal

from ._models import GigaamV2Ctc, GigaamV2Rnnt, KaldiTransducer, NemoConformerCtc, NemoConformerRnnt
from .asr import Asr


def load_model(
    model_type: Literal["gigaam_ctc", "gigaam_rnnt", "kaldi_t", "nemo_ctc", "nemo_rnnt", "vosk"],
    model_path: str | Path,
    int8: bool = False,
) -> Asr:
    """Load ASR model.

    Args:
        model_type: Select model type:
                    GigaAM v2 (`gigaam_ctc` for CTC version or `gigaam_rnnt` for RNN-T),
                    Kaldi Transducer (`kaldi_t` or `vosk`)
                    Nvidia Conformer (`nemo_ctc` for CTC version or `nemo_rnnt` for RNN-T).
        model_path: Path to model files (see README for examples).
        int8: Use int8 model version.

    Returns:
        ASR model class.

    """
    match model_type:
        case "gigaam_ctc":
            model_class = GigaamV2Ctc
        case "gigaam_rnnt":
            model_class = GigaamV2Rnnt
        case "kaldi_t" | "vosk":
            model_class = KaldiTransducer
        case "nemo_ctc":
            model_class = NemoConformerCtc
        case "nemo_rnnt":
            model_class = NemoConformerRnnt

    if Path(model_path).suffix == ".onnx":
        model_name = Path(model_path).stem
        model_path = Path(model_path).parent
    else:
        model_name = None

    def find(part: Path):
        if model_name:
            part = Path(str(part).replace("{model_name}", model_name))

        if int8 and part.suffix == ".onnx":
            part = part.with_suffix(".int8.onnx")

        if Path(model_path, part).exists():
            return Path(model_path, part)

        files = list(Path(model_path).rglob(str(part)))
        assert len(files) == 1, f"File '{part}' not found in path '{model_path}'"
        return files[0]

    return model_class({x[0]: find(Path(x[1])) for x in model_class._get_model_parts().items()})
