import numpy as np
import pytest

import onnx_asr
import onnx_asr.utils
from onnx_asr.adapters import TextResultsAsrAdapter
from onnx_asr.vad import Vad

models = [
    "gigaam-v2-ctc",
    "gigaam-v2-rnnt",
    "nemo-fastconformer-ru-ctc",
    "nemo-fastconformer-ru-rnnt",
    "istupakov/canary-180m-flash-onnx",
    "alphacep/vosk-model-ru",
    "alphacep/vosk-model-small-ru",
    "t-tech/t-one",
    "whisper-base",
    "onnx-community/whisper-tiny",
]


@pytest.fixture(scope="module", params=models)
def model(request: pytest.FixtureRequest) -> TextResultsAsrAdapter:
    if request.param == "t-tech/t-one":
        quantization = None
    elif request.param == "onnx-community/whisper-tiny":
        quantization = "uint8"
    else:
        quantization = "int8"

    return onnx_asr.load_model(request.param, quantization=quantization, providers=["CPUExecutionProvider"])


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
