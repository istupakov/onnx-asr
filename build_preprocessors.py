import preprocessors
from pathlib import Path

if __name__ == "__main__":
    preprocessors_dir = Path("src/onnx_asr/preprocessors")
    preprocessors.save_model(preprocessors.KaldiPreprocessor, preprocessors_dir.joinpath("kaldi.onnx"))
    preprocessors.save_model(preprocessors.GigaamPreprocessor, preprocessors_dir.joinpath("gigaam.onnx"))
    preprocessors.save_model(preprocessors.NemoPreprocessor, preprocessors_dir.joinpath("nemo.onnx"))
    preprocessors.save_model(preprocessors.WhisperPreprocessor, preprocessors_dir.joinpath("whisper.onnx"))
