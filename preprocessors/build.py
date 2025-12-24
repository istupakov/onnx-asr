from pathlib import Path

import onnx
import onnxscript

import preprocessors


def save_model(function: onnxscript.OnnxFunction, filename: Path, input_size_limit: int = 1024 * 1024):
    model = function.to_model_proto()

    model = onnxscript.optimizer.optimize(model, input_size_limit=input_size_limit)

    model.producer_name = "ONNX Script"
    model.producer_version = onnxscript.__version__
    model.metadata_props.add(key="model_author", value="Ilya Stupakov")
    model.metadata_props.add(key="model_license", value="MIT License")

    onnx.checker.check_model(model, full_check=True)
    onnx.save_model(model, filename)


def build():
    preprocessors_dir = Path("src/onnx_asr/preprocessors")
    save_model(preprocessors.KaldiPreprocessor, preprocessors_dir.joinpath("kaldi.onnx"))
    save_model(preprocessors.KaldiPreprocessorFast, preprocessors_dir.joinpath("kaldi_fast.onnx"))
    save_model(preprocessors.GigaamPreprocessorV2, preprocessors_dir.joinpath("gigaam_v2.onnx"))
    save_model(preprocessors.GigaamPreprocessorV3, preprocessors_dir.joinpath("gigaam_v3.onnx"))
    save_model(preprocessors.NemoPreprocessor80, preprocessors_dir.joinpath("nemo80.onnx"))
    save_model(preprocessors.NemoPreprocessor128, preprocessors_dir.joinpath("nemo128.onnx"))
    save_model(preprocessors.WhisperPreprocessor80, preprocessors_dir.joinpath("whisper80.onnx"))
    save_model(preprocessors.WhisperPreprocessor128, preprocessors_dir.joinpath("whisper128.onnx"))

    for orig_freq in [8_000, 11_025, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000]:
        for new_freq in [8_000, 16_000]:
            if orig_freq != new_freq:
                save_model(
                    preprocessors.create_resampler(orig_freq, new_freq),
                    preprocessors_dir.joinpath(f"resample_{orig_freq // 1000}_{new_freq // 1000}.onnx"),
                    100,
                )


if __name__ == "__main__":
    build()
