import numpy as np
import pytest
import torch
import torchaudio

from onnx_asr.preprocessors.preprocessor import ConcurrentPreprocessor, OnnxPreprocessor
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
    return torch.log(
        transform(torch.from_numpy(waveforms)).clamp_(gigaam.clamp_min, gigaam.clamp_max)
    ).numpy(), features_lens


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
        torch.from_numpy(lens - gigaam.win_length_v3)
        .div(gigaam.hop_length, rounding_mode="floor")
        .add(1)
        .long()
        .numpy()
    )
    return torch.log(
        transform(torch.from_numpy(waveforms)).clamp_(gigaam.clamp_min, gigaam.clamp_max)
    ).numpy(), features_lens


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
    mel_spectrogram = torch.matmul(
        spectrogram.transpose(-1, -2), torch.from_numpy(gigaam.melscale_fbanks_v2)
    ).transpose(-1, -2)
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
    mel_spectrogram = torch.matmul(
        spectrogram.transpose(-1, -2), torch.from_numpy(gigaam.melscale_fbanks_v3)
    ).transpose(-1, -2)
    return torch.log(mel_spectrogram.clamp_(gigaam.clamp_min, gigaam.clamp_max)).numpy(), (
        lens - gigaam.win_length_v3
    ) // gigaam.hop_length + 1


@pytest.fixture(scope="module", params=["torch", "onnx_func", "onnx_model", "onnx_model_mt"])
def preprocessor_v2(request):
    match request.param:
        case "torch":
            return (preprocessor_torch_v2, True)
        case "onnx_func":
            return (gigaam.GigaamPreprocessorV2, False)
        case "onnx_model":
            return (OnnxPreprocessor("gigaam_v2", {}), False)
        case "onnx_model_mt":
            return (ConcurrentPreprocessor(OnnxPreprocessor("gigaam_v2", {}), 2), False)


@pytest.fixture(scope="module", params=["torch", "onnx_func", "onnx_model", "onnx_model_mt"])
def preprocessor_v3(request):
    match request.param:
        case "torch":
            return (preprocessor_torch_v3, True)
        case "onnx_func":
            return (gigaam.GigaamPreprocessorV3, False)
        case "onnx_model":
            return (OnnxPreprocessor("gigaam_v3", {}), False)
        case "onnx_model_mt":
            return (ConcurrentPreprocessor(OnnxPreprocessor("gigaam_v3", {}), 2), False)


def test_gigaam_preprocessor_v2(preprocessor_v2, waveforms):
    preprocessor, equal = preprocessor_v2
    waveforms, lens = pad_list(waveforms)
    expected, expected_lens = preprocessor_origin_v2(waveforms, lens)
    actual, actual_lens = preprocessor(waveforms, lens)

    assert expected.shape[2] == max(expected_lens)
    np.testing.assert_equal(actual_lens, expected_lens)
    if equal:
        np.testing.assert_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, atol=5e-5)


def test_gigaam_preprocessor_v3(preprocessor_v3, waveforms):
    preprocessor, equal = preprocessor_v3
    waveforms, lens = pad_list(waveforms)
    expected, expected_lens = preprocessor_origin_v3(waveforms, lens)
    actual, actual_lens = preprocessor(waveforms, lens)

    assert expected.shape[2] == max(expected_lens)
    np.testing.assert_equal(actual_lens, expected_lens)
    if equal:
        np.testing.assert_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, atol=5e-5, rtol=5e-6)
