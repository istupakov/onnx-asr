import numpy as np
import pytest

import onnx_asr
import onnx_asr.utils
from onnx_asr.adapters import TextResultsAsrAdapter

models = [
    "gigaam-v2-ctc",
    "gigaam-v2-rnnt",
    "nemo-fastconformer-ru-ctc",
    "nemo-fastconformer-ru-rnnt",
    "alphacep/vosk-model-ru",
    "alphacep/vosk-model-small-ru",
    "whisper-base",
    "onnx-community/whisper-tiny",
]


@pytest.fixture(scope="module")
def model(request: pytest.FixtureRequest) -> TextResultsAsrAdapter:
    return onnx_asr.load_model(request.param, quantization="int8" if request.param != "onnx-community/whisper-tiny" else "uint8")


@pytest.mark.parametrize("model", models, indirect=True)
def test_supported_only_mono_audio_error(model: TextResultsAsrAdapter) -> None:
    rng = np.random.default_rng(0)
    waveform = rng.random((1 * 16_000, 2), dtype=np.float32)

    with pytest.raises(onnx_asr.utils.SupportedOnlyMonoAudioError):
        model.recognize(waveform)


@pytest.mark.parametrize("model", models, indirect=True)
def test_wrong_sample_rate_error(model: TextResultsAsrAdapter) -> None:
    rng = np.random.default_rng(0)
    waveform = rng.random((1 * 16_000), dtype=np.float32)

    with pytest.raises(onnx_asr.utils.WrongSampleRateError):
        model.recognize(waveform, sample_rate=24_000)  # type: ignore


@pytest.mark.parametrize("model", models, indirect=True)
def test_recognize(model: TextResultsAsrAdapter) -> None:
    rng = np.random.default_rng(0)
    waveform = rng.random((1 * 16_000), dtype=np.float32)

    result = model.recognize(waveform)
    assert isinstance(result, str)
