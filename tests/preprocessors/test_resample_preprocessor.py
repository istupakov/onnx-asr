import numpy as np
import pytest
import torch
import torchaudio

from onnx_asr.preprocessors import Resampler
from onnx_asr.utils import pad_list
from preprocessors import resample


def onnx_preprocessor(waveforms, waveforms_lens, sample_rate):
    match sample_rate:
        case 8_000:
            return resample.ResamplePreprocessor8(waveforms, waveforms_lens)
        case 16_000:
            return waveforms, waveforms_lens
        case 22_050:
            return resample.ResamplePreprocessor22(waveforms, waveforms_lens)
        case 24_000:
            return resample.ResamplePreprocessor24(waveforms, waveforms_lens)
        case 32_000:
            return resample.ResamplePreprocessor32(waveforms, waveforms_lens)
        case 44_100:
            return resample.ResamplePreprocessor44(waveforms, waveforms_lens)
        case 48_000:
            return resample.ResamplePreprocessor48(waveforms, waveforms_lens)


@pytest.fixture(scope="module")
def preprocessor(request):
    match request.param:
        case "onnx_func":
            return onnx_preprocessor
        case "onnx_model":
            return Resampler({})


@pytest.mark.parametrize(
    "preprocessor",
    [
        "onnx_func",
        "onnx_model",
    ],
    indirect=True,
)
@pytest.mark.parametrize("sample_rate", [8_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000])
def test_resample_preprocessor(preprocessor, sample_rate, waveforms):
    expected = [
        torchaudio.functional.resample(torch.tensor(waveform).unsqueeze(0), sample_rate, 16_000)[0].numpy()
        for waveform in waveforms
    ]
    expected, expected_lens = pad_list(expected)
    actual, actual_lens = preprocessor(*pad_list(waveforms), sample_rate)

    np.testing.assert_equal(actual_lens, expected_lens)
    np.testing.assert_allclose(actual, expected, atol=1e-6)
