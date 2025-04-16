import onnxscript
from pathlib import Path

import preprocessors


def save_model(function: onnxscript.OnnxFunction, filename: str):
    model = function.to_model_proto()
    model = onnxscript.optimizer.optimize(model)
    model = onnxscript.ir.from_proto(model)
    model = onnxscript.optimizer.optimize(model)

    model.producer_name = "OnnxScript"
    model.producer_version = onnxscript.__version__
    model.metadata_props["model_author"] = "Ilya Stupakov"
    model.metadata_props["model_license"] = "MIT License"

    onnxscript.ir.save(model, filename)


if __name__ == "__main__":
    preprocessors_dir = Path("src/onnx_asr/preprocessors")
    save_model(preprocessors.KaldiPreprocessor, preprocessors_dir.joinpath("kaldi.onnx"))
    save_model(preprocessors.GigaamPreprocessor, preprocessors_dir.joinpath("gigaam.onnx"))
    save_model(preprocessors.NemoPreprocessor, preprocessors_dir.joinpath("nemo.onnx"))
    save_model(preprocessors.WhisperPreprocessor, preprocessors_dir.joinpath("whisper.onnx"))
