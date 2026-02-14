from collections.abc import Iterator

import numpy as np
import onnxruntime
import pytest

import onnx_asr
import onnx_asr.utils
from onnx_asr.adapters import TextResultsAsrAdapter
from onnx_asr.asr import BaseAsr, TimestampedResult
from onnx_asr.preprocessors.numpy_preprocessor import _NumpyPreprocessor
from onnx_asr.preprocessors.preprocessor import ConcurrentPreprocessor, OnnxPreprocessor
from onnx_asr.vad import BaseVad, SegmentResult, TimestampedSegmentResult, Vad

models = [
    "gigaam-v2-ctc",
    "gigaam-v2-rnnt",
    "nemo-fastconformer-ru-ctc",
    "nemo-fastconformer-ru-rnnt",
    "alphacep/vosk-model-ru",
    "alphacep/vosk-model-small-ru",
    "t-tech/t-one",
    "whisper-base",
    "onnx-community/whisper-tiny",
    pytest.param(
        "istupakov/canary-180m-flash-onnx",
        marks=pytest.mark.xfail(onnxruntime.__version__ == "1.18.1", reason="Missed Trilu ONNX operator"),
    ),
]


@pytest.fixture(scope="module", params=models)
def model(request: pytest.FixtureRequest) -> TextResultsAsrAdapter:
    match request.param:
        case "t-tech/t-one":
            return onnx_asr.load_model(request.param)
        case "onnx-community/whisper-tiny":
            return onnx_asr.load_model(request.param, quantization="uint8")
        case _:
            return onnx_asr.load_model(request.param, quantization="int8")


@pytest.fixture(scope="module")
def vad() -> Vad:
    return onnx_asr.load_vad("silero")


def test_file_not_found_error(model: TextResultsAsrAdapter) -> None:
    with pytest.raises(FileNotFoundError):
        model.recognize("test.wav")


def test_supported_only_mono_audio_error(model: TextResultsAsrAdapter) -> None:
    rng = np.random.default_rng(0)
    waveform = rng.random((1 * 16_000, 2), dtype=np.float32)

    with pytest.raises(onnx_asr.utils.SupportedOnlyMonoAudioError):
        model.recognize(waveform)


def test_wrong_sample_rate_error(model: TextResultsAsrAdapter) -> None:
    rng = np.random.default_rng(0)
    waveform = rng.random((1 * 16_000), dtype=np.float32)

    with pytest.raises(onnx_asr.utils.WrongSampleRateError):
        model.recognize(waveform, sample_rate=25_000)  # type: ignore


def test_recognize(model: TextResultsAsrAdapter) -> None:
    rng = np.random.default_rng(0)
    waveform = rng.random((1 * 16_000), dtype=np.float32)

    result = model.recognize(waveform)
    assert isinstance(result, str)


def test_empty_recognize(model: TextResultsAsrAdapter) -> None:
    result = model.recognize([])
    assert result == []


def test_recognize_with_timestamps(model: TextResultsAsrAdapter) -> None:
    rng = np.random.default_rng(0)
    waveform = rng.random((1 * 16_000), dtype=np.float32)

    result = model.with_timestamps().recognize(waveform)
    assert isinstance(result, TimestampedResult)


def test_recognize_batch(model: TextResultsAsrAdapter) -> None:
    rng = np.random.default_rng(0)
    waveform1 = rng.random((2 * 16_000), dtype=np.float32)
    waveform2 = rng.random((1 * 16_000), dtype=np.float32)

    result = model.recognize([waveform1, waveform2])
    assert isinstance(result, list)
    assert all(isinstance(item, str) for item in result)


def test_recognize_with_vad(model: TextResultsAsrAdapter, vad: BaseVad) -> None:
    rng = np.random.default_rng(0)
    waveform = rng.random((1 * 16_000), dtype=np.float32)

    result = model.with_vad(vad).recognize(waveform)
    assert isinstance(result, Iterator)
    assert all(isinstance(item, SegmentResult) for item in result)


def test_recognize_with_vad_and_timestamps(model: TextResultsAsrAdapter, vad: BaseVad) -> None:
    rng = np.random.default_rng(0)
    waveform = rng.random((1 * 16_000), dtype=np.float32)

    result = model.with_vad(vad).with_timestamps().recognize(waveform)
    assert isinstance(result, Iterator)
    assert all(isinstance(item, TimestampedSegmentResult) for item in result)


@pytest.mark.parametrize("max_concurrent_workers", [None, 1, 2])
@pytest.mark.parametrize("use_numpy_preprocessors", [None, True, False])
def test_preprocessor_options(max_concurrent_workers: int | None, use_numpy_preprocessors: bool | None) -> None:
    model = onnx_asr.load_model(
        "alphacep/vosk-model-small-ru",
        quantization="int8",
        preprocessor_config={
            "max_concurrent_workers": max_concurrent_workers,
            "use_numpy_preprocessors": use_numpy_preprocessors,
        },
    )
    rng = np.random.default_rng(0)
    waveform = rng.random((1 * 16_000), dtype=np.float32)

    asr = model.asr
    assert isinstance(asr, BaseAsr)
    if max_concurrent_workers != 1:
        assert isinstance(asr._preprocessor, ConcurrentPreprocessor)
    elif use_numpy_preprocessors is True:
        assert isinstance(asr._preprocessor, _NumpyPreprocessor)
    elif use_numpy_preprocessors is False:
        assert isinstance(asr._preprocessor, OnnxPreprocessor)

    result = model.recognize([waveform] * 2)
    assert isinstance(result, list)
    assert all(isinstance(item, str) for item in result)
