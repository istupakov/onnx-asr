from pathlib import Path

from preprocessors import build


def test_preprocessors_build(tmp_path: Path):
    build.build_models(tmp_path, "tests")
    assert len(list(tmp_path.glob("*.onnx"))) == 22
