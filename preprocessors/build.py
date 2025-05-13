from pathlib import Path

import onnx
import onnxscript

import preprocessors


def save_model(function: onnxscript.OnnxFunction, filename: Path):
    model = function.to_model_proto()
    model = onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=True, data_prop=True)
    onnx.checker.check_model(model, full_check=True)

    model = onnxscript.ir.serde.deserialize_model(model)
    model = onnxscript.optimizer.optimize(model)

    model.producer_name = "OnnxScript"
    model.producer_version = onnxscript.__version__
    model.metadata_props["model_author"] = "Ilya Stupakov"
    model.metadata_props["model_license"] = "MIT License"

    model = onnxscript.ir.serde.serialize_model(model)
    onnx.save_model(model, filename)


def build():
    preprocessors_dir = Path("src/onnx_asr/preprocessors")
    save_model(preprocessors.KaldiPreprocessor, preprocessors_dir.joinpath("kaldi.onnx"))
    save_model(preprocessors.GigaamPreprocessor, preprocessors_dir.joinpath("gigaam.onnx"))
    save_model(preprocessors.NemoPreprocessor80, preprocessors_dir.joinpath("nemo80.onnx"))
    save_model(preprocessors.NemoPreprocessor128, preprocessors_dir.joinpath("nemo128.onnx"))
    save_model(preprocessors.WhisperPreprocessor80, preprocessors_dir.joinpath("whisper80.onnx"))
    save_model(preprocessors.WhisperPreprocessor128, preprocessors_dir.joinpath("whisper128.onnx"))
    save_model(preprocessors.ResamplePreprocessor, preprocessors_dir.joinpath("resample.onnx"))
