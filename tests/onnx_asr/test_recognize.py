import numpy as np
import pytest

import onnx_asr
import onnx_asr.utils


@pytest.fixture(scope="module")
def model(request):
    return onnx_asr.load_model(request.param)


@pytest.mark.parametrize("model", ["alphacep/vosk-model-small-ru", "onnx-community/whisper-tiny", "whisper-base"], indirect=True)
def test_supported_only_mono_audio_error(model):
    rng = np.random.default_rng(0)
    waveform = rng.random((1 * 16_000, 2), dtype=np.float32)

    with pytest.raises(onnx_asr.utils.SupportedOnlyMonoAudioError):
        model.recognize(waveform)


@pytest.mark.parametrize("model", ["alphacep/vosk-model-small-ru", "onnx-community/whisper-tiny", "whisper-base"], indirect=True)
def test_wrong_sample_rate_error(model):
    rng = np.random.default_rng(0)
    waveform = rng.random((1 * 16_000), dtype=np.float32)

    with pytest.raises(onnx_asr.utils.WrongSampleRateError):
        model.recognize(waveform, sample_rate=24_000)  # type: ignore


@pytest.mark.parametrize("model", ["alphacep/vosk-model-small-ru", "onnx-community/whisper-tiny", "whisper-base"], indirect=True)
def test_recognize(model):
    rng = np.random.default_rng(0)
    waveform = rng.random((1 * 16_000), dtype=np.float32)

    result = model.recognize(waveform)
    assert isinstance(result, str)
