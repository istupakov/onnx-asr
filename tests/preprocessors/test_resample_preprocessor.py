import numpy as np
import pytest
import torch
import torchaudio

from onnx_asr.preprocessors import Resampler
from onnx_asr.utils import pad_list
from preprocessors import resample


@pytest.fixture(scope="module")
def preprocessor(request):
    match request.param:
        case "onnx_func":
            return lambda x, x_len, sr: resample.ResamplePreprocessor(x, x_len, [sr])
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
@pytest.mark.parametrize("sample_rate", [8_000, 16_000, 22_050, 44_100, 48_000])
def test_resample_preprocessor(preprocessor, sample_rate, waveforms):
    expected = [
        torchaudio.functional.resample(torch.tensor(waveform).unsqueeze(0), sample_rate, 16_000)[0].numpy()
        for waveform in waveforms
    ]
    expected, expected_lens = pad_list(expected)
    actual, actual_lens = preprocessor(*pad_list(waveforms), sample_rate)

    np.testing.assert_equal(actual_lens, expected_lens)
    np.testing.assert_allclose(actual, expected, atol=1e-6)
