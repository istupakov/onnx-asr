import numpy as np
import pytest
import torch
import torchaudio

from onnx_asr.preprocessors import Preprocessor
from onnx_asr.utils import pad_list
from preprocessors import gigaam


def preprocessor_origin(waveforms, lens):
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=gigaam.sample_rate,
        n_fft=gigaam.n_fft,
        win_length=gigaam.win_length,
        hop_length=gigaam.hop_length,
        n_mels=gigaam.n_mels,
    )
    features_lens = torch.from_numpy(lens).div(gigaam.hop_length, rounding_mode="floor").add(1).long().numpy()
    return torch.log(transform(torch.from_numpy(waveforms)).clamp_(gigaam.clamp_min, gigaam.clamp_max)).numpy(), features_lens


def preprocessor_torch(waveforms, lens):
    waveforms = torch.from_numpy(waveforms)
    spectrogram = torchaudio.functional.spectrogram(
        waveforms,
        pad=0,
        window=torch.hann_window(gigaam.win_length),
        n_fft=gigaam.win_length,
        hop_length=gigaam.hop_length,
        win_length=gigaam.win_length,
        power=2,
        normalized=False,
    )
    mel_spectrogram = torch.matmul(spectrogram.transpose(-1, -2), gigaam.melscale_fbanks).transpose(-1, -2)
    return torch.log(mel_spectrogram.clamp_(gigaam.clamp_min, gigaam.clamp_max)).numpy(), lens // gigaam.hop_length + 1


@pytest.fixture(scope="module")
def preprocessor(request):
    match request.param:
        case "torch":
            return preprocessor_torch
        case "onnx_func":
            return gigaam.GigaamPreprocessor
        case "onnx_model":
            return Preprocessor("gigaam", {})


@pytest.mark.parametrize(
    "preprocessor,equal",
    [
        ("torch", True),
        ("onnx_func", False),
        ("onnx_model", False),
    ],
    indirect=["preprocessor"],
)
def test_gigaam_preprocessor(preprocessor, equal, waveforms):
    waveforms, lens = pad_list(waveforms)
    expected, expected_lens = preprocessor_origin(waveforms, lens)
    actual, actual_lens = preprocessor(waveforms, lens)

    assert expected.shape[2] == max(expected_lens)
    np.testing.assert_equal(actual_lens, expected_lens)
    if equal:
        np.testing.assert_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, atol=5e-5)
