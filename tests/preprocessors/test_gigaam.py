import numpy as np
import pytest
import torch
import torchaudio

from onnx_asr.preprocessors import Preprocessor
from onnx_asr.utils import pad_list
from preprocessors import gigaam


def preprocessor_origin_v2(waveforms, lens):
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=gigaam.sample_rate,
        n_fft=gigaam.n_fft_v2,
        win_length=gigaam.win_length_v2,
        hop_length=gigaam.hop_length,
        n_mels=gigaam.n_mels,
    )
    features_lens = torch.from_numpy(lens).div(gigaam.hop_length, rounding_mode="floor").add(1).long().numpy()
    return torch.log(transform(torch.from_numpy(waveforms)).clamp_(gigaam.clamp_min, gigaam.clamp_max)).numpy(), features_lens


def preprocessor_origin_v3(waveforms, lens):
    transform = (
        torchaudio.transforms.MelSpectrogram(
            sample_rate=gigaam.sample_rate,
            n_fft=gigaam.n_fft_v3,
            win_length=gigaam.win_length_v3,
            hop_length=gigaam.hop_length,
            n_mels=gigaam.n_mels,
            center=False,
        )
        .bfloat16()
        .float()
    )
    features_lens = (
        torch.from_numpy(lens - gigaam.win_length_v3).div(gigaam.hop_length, rounding_mode="floor").add(1).long().numpy()
    )
    return torch.log(transform(torch.from_numpy(waveforms)).clamp_(gigaam.clamp_min, gigaam.clamp_max)).numpy(), features_lens


def preprocessor_torch_v2(waveforms, lens):
    waveforms = torch.from_numpy(waveforms)
    spectrogram = torchaudio.functional.spectrogram(
        waveforms,
        pad=0,
        window=torch.hann_window(gigaam.win_length_v2),
        n_fft=gigaam.n_fft_v2,
        hop_length=gigaam.hop_length,
        win_length=gigaam.win_length_v2,
        power=2,
        normalized=False,
    )
    mel_spectrogram = torch.matmul(spectrogram.transpose(-1, -2), torch.from_numpy(gigaam.melscale_fbanks_v2)).transpose(-1, -2)
    return torch.log(mel_spectrogram.clamp_(gigaam.clamp_min, gigaam.clamp_max)).numpy(), lens // gigaam.hop_length + 1


def preprocessor_torch_v3(waveforms, lens):
    waveforms = torch.from_numpy(waveforms)
    spectrogram = torchaudio.functional.spectrogram(
        waveforms,
        pad=0,
        window=torch.hann_window(gigaam.win_length_v3).bfloat16().float(),
        n_fft=gigaam.n_fft_v3,
        hop_length=gigaam.hop_length,
        win_length=gigaam.win_length_v3,
        power=2,
        normalized=False,
        center=False,
    )
    mel_spectrogram = torch.matmul(spectrogram.transpose(-1, -2), torch.from_numpy(gigaam.melscale_fbanks_v3)).transpose(-1, -2)
    return torch.log(mel_spectrogram.clamp_(gigaam.clamp_min, gigaam.clamp_max)).numpy(), (
        lens - gigaam.win_length_v3
    ) // gigaam.hop_length + 1


@pytest.fixture(scope="module")
def preprocessor(request):
    match request.param:
        case "torch_v2":
            return preprocessor_torch_v2
        case "onnx_func_v2":
            return gigaam.GigaamPreprocessorV2
        case "onnx_model_v2":
            return Preprocessor("gigaam_v2", {})
        case "onnx_model_v2_mt":
            return Preprocessor("gigaam_v2", {"max_concurrent_workers": 2})
        case "torch_v3":
            return preprocessor_torch_v3
        case "onnx_func_v3":
            return gigaam.GigaamPreprocessorV3
        case "onnx_model_v3":
            return Preprocessor("gigaam_v3", {})
        case "onnx_model_v3_mt":
            return Preprocessor("gigaam_v3", {"max_concurrent_workers": 2})


@pytest.mark.parametrize(
    ("preprocessor", "equal"),
    [
        ("torch_v2", True),
        ("onnx_func_v2", False),
        ("onnx_model_v2", False),
        ("onnx_model_v2_mt", False),
    ],
    indirect=["preprocessor"],
)
def test_gigaam_preprocessor_v2(preprocessor, equal, waveforms):
    waveforms, lens = pad_list(waveforms)
    expected, expected_lens = preprocessor_origin_v2(waveforms, lens)
    actual, actual_lens = preprocessor(waveforms, lens)

    assert expected.shape[2] == max(expected_lens)
    np.testing.assert_equal(actual_lens, expected_lens)
    if equal:
        np.testing.assert_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, atol=5e-5)


@pytest.mark.parametrize(
    ("preprocessor", "equal"),
    [
        ("torch_v3", True),
        ("onnx_func_v3", False),
        ("onnx_model_v3", False),
        ("onnx_model_v3_mt", False),
    ],
    indirect=["preprocessor"],
)
def test_gigaam_preprocessor_v3(preprocessor, equal, waveforms):
    waveforms, lens = pad_list(waveforms)
    expected, expected_lens = preprocessor_origin_v3(waveforms, lens)
    actual, actual_lens = preprocessor(waveforms, lens)

    assert expected.shape[2] == max(expected_lens)
    np.testing.assert_equal(actual_lens, expected_lens)
    if equal:
        np.testing.assert_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, atol=5e-5, rtol=5e-6)
