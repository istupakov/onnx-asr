from pathlib import Path

import onnx
import onnxscript

import preprocessors


def save_model(function: onnxscript.OnnxFunction, filename: Path):
    model = function.to_model_proto()

    model = onnxscript.optimizer.optimize(model, input_size_limit=100)

    model.producer_name = "OnnxScript"
    model.producer_version = onnxscript.__version__
    model.metadata_props.add(key="model_author", value="Ilya Stupakov")
    model.metadata_props.add(key="model_license", value="MIT License")

    onnx.checker.check_model(model, full_check=True)
    onnx.save_model(model, filename)


def build():
    preprocessors_dir = Path("src/onnx_asr/preprocessors")
    save_model(preprocessors.KaldiPreprocessor, preprocessors_dir.joinpath("kaldi.onnx"))
    save_model(preprocessors.GigaamPreprocessorV2, preprocessors_dir.joinpath("gigaam_v2.onnx"))
    save_model(preprocessors.GigaamPreprocessorV3, preprocessors_dir.joinpath("gigaam_v3.onnx"))
    save_model(preprocessors.NemoPreprocessor80, preprocessors_dir.joinpath("nemo80.onnx"))
    save_model(preprocessors.NemoPreprocessor128, preprocessors_dir.joinpath("nemo128.onnx"))
    save_model(preprocessors.WhisperPreprocessor80, preprocessors_dir.joinpath("whisper80.onnx"))
    save_model(preprocessors.WhisperPreprocessor128, preprocessors_dir.joinpath("whisper128.onnx"))
    save_model(preprocessors.ResamplePreprocessor8, preprocessors_dir.joinpath("resample8.onnx"))
    save_model(preprocessors.ResamplePreprocessor16, preprocessors_dir.joinpath("resample16.onnx"))
