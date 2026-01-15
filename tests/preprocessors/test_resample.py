from typing import get_args

import numpy as np
import pytest
import torch
import torchaudio

from onnx_asr.preprocessors import Resampler
from onnx_asr.utils import SampleRates, pad_list
from preprocessors import resample


def onnx_preprocessor8(waveforms, waveforms_lens, sample_rate):
    if sample_rate == 8_000:
        return waveforms, waveforms_lens
    return resample.create_resampler(sample_rate, 8_000)(waveforms, waveforms_lens)


def onnx_preprocessor16(waveforms, waveforms_lens, sample_rate):
    if sample_rate == 16_000:
        return waveforms, waveforms_lens
    return resample.create_resampler(sample_rate, 16_000)(waveforms, waveforms_lens)


@pytest.fixture(scope="module")
def preprocessor(request):
    match request.param:
        case "onnx_func8":
            return onnx_preprocessor8
        case "onnx_func16":
            return onnx_preprocessor16
        case "onnx_model8":
            return Resampler(8_000, {})
        case "onnx_model16":
            return Resampler(16_000, {})


@pytest.mark.parametrize(
    "preprocessor",
    [
        "onnx_func8",
        "onnx_model8",
    ],
    indirect=True,
)
@pytest.mark.parametrize("sample_rate", [8_000, 11_025, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000])
def test_resample8_preprocessor(preprocessor, sample_rate, waveforms):
    expected = [
        torchaudio.functional.resample(torch.tensor(waveform).unsqueeze(0), sample_rate, 8_000)[0].numpy()
        for waveform in waveforms
    ]
    expected, expected_lens = pad_list(expected)
    actual, actual_lens = preprocessor(*pad_list(waveforms), sample_rate)

    np.testing.assert_equal(actual_lens, expected_lens)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize(
    "preprocessor",
    [
        "onnx_func16",
        "onnx_model16",
    ],
    indirect=True,
)
@pytest.mark.parametrize("sample_rate", get_args(SampleRates))
def test_resample16_preprocessor(preprocessor, sample_rate, waveforms):
    expected = [
        torchaudio.functional.resample(torch.tensor(waveform).unsqueeze(0), sample_rate, 16_000)[0].numpy()
        for waveform in waveforms
    ]
    expected, expected_lens = pad_list(expected)
    actual, actual_lens = preprocessor(*pad_list(waveforms), sample_rate)

    np.testing.assert_equal(actual_lens, expected_lens)
    np.testing.assert_allclose(actual, expected, atol=1e-6)
