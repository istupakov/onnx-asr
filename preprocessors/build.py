"""Build ONNX preprocessor models."""

from pathlib import Path

import onnx
import onnxscript

from preprocessors import gigaam, kaldi, nemo, resample, whisper


def build_model(function: onnxscript.OnnxFunction, filename: Path, version: str, input_size_limit: int = 1024 * 1024):
    model = function.to_model_proto()

    model = onnxscript.optimizer.optimize(model, input_size_limit=input_size_limit)

    model.producer_name = "OnnxScript"
    model.producer_version = onnxscript.__version__
    model.metadata_props.add(key="model_author", value="Ilya Stupakov")
    model.metadata_props.add(key="model_license", value="MIT License")
    model.metadata_props.add(key="model_version", value=f"onnx-asr {version}")

    onnx.checker.check_model(model, full_check=True)
    onnx.save_model(model, filename)


def build_models(preprocessors_dir: Path, version: str):
    preprocessors = {
        "gigaam_v2.onnx": gigaam.GigaamPreprocessorV2,
        "gigaam_v3.onnx": gigaam.GigaamPreprocessorV3,
        "kaldi.onnx": kaldi.KaldiPreprocessor,
        "kaldi_fast.onnx": kaldi.KaldiPreprocessorFast,
        "nemo80.onnx": nemo.NemoPreprocessor80,
        "nemo128.onnx": nemo.NemoPreprocessor128,
        "whisper80.onnx": whisper.WhisperPreprocessor80,
        "whisper128.onnx": whisper.WhisperPreprocessor128,
    }
    for filename, model in preprocessors.items():
        build_model(model, preprocessors_dir.joinpath(filename), version)

    for orig_freq in [8_000, 11_025, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000]:
        for new_freq in [8_000, 16_000]:
            if orig_freq != new_freq:
                build_model(
                    resample.create_resampler(orig_freq, new_freq),
                    preprocessors_dir.joinpath(f"resample_{orig_freq // 1000}_{new_freq // 1000}.onnx"),
                    version,
                    100,
                )
