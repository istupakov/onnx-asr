import numpy as np
import onnxruntime
import pytest

import onnx_asr
import onnx_asr.utils
from onnx_asr.adapters import TextResultsAsrAdapter
from onnx_asr.asr import TimestampedResult
from onnx_asr.vad import Vad

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
        marks=pytest.mark.skipif(onnxruntime.__version__ == "1.18.1", reason="Missed Trilu ONNX operator"),
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


def test_recognize_with_vad(model: TextResultsAsrAdapter, vad: Vad) -> None:
    rng = np.random.default_rng(0)
    waveform = rng.random((1 * 16_000), dtype=np.float32)

    result = list(model.with_vad(vad).recognize(waveform))
    assert isinstance(result, list)


def test_concurrent_preprocessor() -> None:
    model = onnx_asr.load_model(
        "alphacep/vosk-model-small-ru", quantization="int8", preprocessor_config={"max_concurrent_workers": 2}
    )
    rng = np.random.default_rng(0)
    waveform = rng.random((1 * 16_000), dtype=np.float32)

    result = model.recognize([waveform] * 2)
    assert isinstance(result, list)
