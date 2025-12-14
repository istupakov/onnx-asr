import pytest

import onnx_asr


def test_model_not_supported_error() -> None:
    with pytest.raises(onnx_asr.loader.ModelNotSupportedError):
        onnx_asr.load_model("xxx")


def test_model_path_not_found_error() -> None:
    with pytest.raises(onnx_asr.loader.ModelPathNotDirectoryError):
        onnx_asr.load_model("whisper", "pyproject.toml")


def test_model_file_not_found_error() -> None:
    with pytest.raises(onnx_asr.loader.ModelFileNotFoundError):
        onnx_asr.load_model("onnx-community/whisper-tiny", quantization="xxx")


def test_more_than_one_model_file_found_error() -> None:
    with pytest.raises(onnx_asr.loader.MoreThanOneModelFileFoundError):
        onnx_asr.load_model("onnx-community/whisper-tiny", quantization="*int8")


def test_no_model_name_or_path_specified_error() -> None:
    with pytest.raises(onnx_asr.loader.NoModelNameOrPathSpecifiedError):
        onnx_asr.load_model("whisper")


def test_no_model_name_and_empty_path_specified_error() -> None:
    with pytest.raises(onnx_asr.loader.NoModelNameOrPathSpecifiedError):
        onnx_asr.load_model("whisper", "./xxx")


def test_invalid_model_type_in_config_error() -> None:
    with pytest.raises(onnx_asr.loader.InvalidModelTypeInConfigError):
        onnx_asr.load_model("onnx-community/pyannote-segmentation-3.0")
