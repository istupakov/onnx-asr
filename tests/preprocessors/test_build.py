from pathlib import Path

import onnx

from preprocessors import build


def test_build(tmp_path: Path):
    build.build(tmp_path, "tests")
    assert len(list(tmp_path.glob("*.onnx"))) == 22


def test_save_preprocessor_models(tmp_path: Path):
    build.save_preprocessor_models(tmp_path, "tests")
    files = list(tmp_path.glob("*.onnx"))

    assert len(files) == 8
    for filename in files:
        onnx.checker.check_model(filename, full_check=True)
        model = onnx.load_model(filename)

        assert len(model.graph.input) == 2
        assert model.graph.input[0].name == "waveforms"
        assert model.graph.input[1].name == "waveforms_lens"

        assert len(model.graph.output) == 2
        assert model.graph.output[0].name == "features"
        assert model.graph.output[1].name == "features_lens"


def test_save_resampler_models(tmp_path: Path):
    build.save_resampler_models(tmp_path, "tests")
    files = list(tmp_path.glob("*.onnx"))

    assert len(files) == 14
    for filename in files:
        onnx.checker.check_model(filename, full_check=True)
        model = onnx.load_model(filename)

        assert len(model.graph.input) == 2
        assert model.graph.input[0].name == "waveforms"
        assert model.graph.input[1].name == "waveforms_lens"

        assert len(model.graph.output) == 2
        assert model.graph.output[0].name == "resampled"
        assert model.graph.output[1].name == "resampled_lens"
