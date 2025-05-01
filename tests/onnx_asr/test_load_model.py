import pytest

import onnx_asr


def test_model_not_supported_error():
    with pytest.raises(onnx_asr.loader.ModelNotSupportedError):
        onnx_asr.load_model("xxx")


def test_model_path_not_found_error():
    with pytest.raises(onnx_asr.loader.ModelPathNotFoundError):
        onnx_asr.load_model("onnx-community/whisper-tiny", "./xxx")


def test_model_file_not_found_error():
    with pytest.raises(onnx_asr.loader.ModelFileNotFoundError):
        onnx_asr.load_model("onnx-community/whisper-tiny", quantization="xxx")


def test_more_than_one_model_file_found_error():
    with pytest.raises(onnx_asr.loader.MoreThanOneModelFileFoundError):
        onnx_asr.load_model("onnx-community/whisper-tiny", quantization="*int8")


def test_no_model_name_or_path_specified_error():
    with pytest.raises(onnx_asr.loader.NoModelNameOrPathSpecifiedError):
        onnx_asr.load_model("whisper-hf")


@pytest.mark.parametrize("model", ["alphacep/vosk-model-small-ru", "onnx-community/whisper-tiny", "whisper-base"])
def test_load_model(model):
    onnx_asr.load_model(model)
