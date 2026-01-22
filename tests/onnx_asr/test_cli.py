from pathlib import Path

import numpy as np
import pytest

from onnx_asr.cli import parse_args, run


@pytest.fixture(params=[True, False], ids=["with-vad", "without-vad"])
def args_list(tmp_path: Path, request: pytest.FixtureRequest) -> list[str]:
    vad = ["--vad", "silero"] if request.param else []
    return [
        "whisper-base",
        str(tmp_path.joinpath("test1.wav")),
        str(tmp_path.joinpath("test2.wav")),
        "-q",
        "int8",
        "--lang",
        "en",
        "-p",
        str(tmp_path.joinpath("model")),
        *vad,
    ]


def test_parse_args(args_list: list[str]) -> None:
    args = parse_args(args_list)
    assert args.model == "whisper-base"
    assert args.filename[0].name == "test1.wav"
    assert args.filename[1].name == "test2.wav"
    assert args.model_path.name == "model"
    assert args.quantization == "int8"
    assert args.lang == "en"
    assert args.vad == ("silero" if "silero" in args_list else None)


def test_file_not_found_error(args_list: list[str]) -> None:
    with pytest.raises(FileNotFoundError):
        run(parse_args(args_list))


def test_cli_run(args_list: list[str]) -> None:
    try:
        import soundfile as sf  # type: ignore[import-untyped]  # noqa: PLC0415
    except ImportError:
        pytest.skip("soundfile not available")
    args = parse_args(args_list)
    args.model_path = None

    rng = np.random.default_rng(0)
    for file in args.filename:
        sf.write(file, rng.random((16_000), dtype=np.float32), 16_000)

    run(args)
