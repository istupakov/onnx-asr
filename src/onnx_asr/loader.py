from pathlib import Path
from typing import Literal

from .asr import Asr
from .models import GigaamV2Ctc, GigaamV2Rnnt, KaldiTransducer, NemoConformerCtc, NemoConformerRnnt


def load_model(
    model: Literal["gigaam_ctc", "gigaam_rnnt", "kaldi_t", "nemo_ctc", "nemo_rnnt"], model_path: str | Path, int8: bool = False
) -> Asr:
    match model:
        case "gigaam_ctc":
            model_type = GigaamV2Ctc
        case "gigaam_rnnt":
            model_type = GigaamV2Rnnt
        case "kaldi_t":
            model_type = KaldiTransducer
        case "nemo_ctc":
            model_type = NemoConformerCtc
        case "nemo_rnnt":
            model_type = NemoConformerRnnt

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

    return model_type(dict(map(lambda x: (x[0], find(Path(x[1]))), model_type._get_model_parts().items())))
