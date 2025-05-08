from pathlib import Path

import onnxscript

import preprocessors


def save_model(function: onnxscript.OnnxFunction, filename: Path):
    model = onnxscript.ir.from_proto(function.to_model_proto())
    model = onnxscript.optimizer.optimize(model)

    model.producer_name = "OnnxScript"
    model.producer_version = onnxscript.__version__
    model.metadata_props["model_author"] = "Ilya Stupakov"
    model.metadata_props["model_license"] = "MIT License"

    onnxscript.ir.save(model, filename)


def build():
    preprocessors_dir = Path("src/onnx_asr/preprocessors")
    save_model(preprocessors.KaldiPreprocessor, preprocessors_dir.joinpath("kaldi.onnx"))
    save_model(preprocessors.GigaamPreprocessor, preprocessors_dir.joinpath("gigaam.onnx"))
    save_model(preprocessors.NemoPreprocessor80, preprocessors_dir.joinpath("nemo80.onnx"))
    save_model(preprocessors.NemoPreprocessor128, preprocessors_dir.joinpath("nemo128.onnx"))
    save_model(preprocessors.WhisperPreprocessor80, preprocessors_dir.joinpath("whisper80.onnx"))
    save_model(preprocessors.WhisperPreprocessor128, preprocessors_dir.joinpath("whisper128.onnx"))
    save_model(preprocessors.ResamplePreprocessor, preprocessors_dir.joinpath("resample.onnx"))
