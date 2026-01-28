from typing import get_args

import numpy as np
import pytest
import torch
import torchaudio

from onnx_asr.preprocessors.resampler import Resampler
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


@pytest.fixture(scope="module", params=[8_000, 16_000])
def target_sample_rate(request):
    return request.param


@pytest.fixture(scope="module", params=["onnx_func", "onnx_model"])
def preprocessor(target_sample_rate, request):
    match request.param:
        case "onnx_func":
            return onnx_preprocessor8 if target_sample_rate == 8_000 else onnx_preprocessor16
        case "onnx_model":
            return Resampler(target_sample_rate, {})


@pytest.mark.parametrize("sample_rate", get_args(SampleRates))
def test_resample_preprocessor(preprocessor, sample_rate, target_sample_rate, waveforms):
    expected = [
        torchaudio.functional.resample(torch.tensor(waveform).unsqueeze(0), sample_rate, target_sample_rate)[0].numpy()
        for waveform in waveforms
    ]
    expected, expected_lens = pad_list(expected)
    actual, actual_lens = preprocessor(*pad_list(waveforms), sample_rate)

    np.testing.assert_equal(actual_lens, expected_lens)
    np.testing.assert_allclose(actual, expected, atol=1e-6)
