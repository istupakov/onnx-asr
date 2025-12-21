import numpy as np
import pytest
import torch
import torchaudio
from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor

from onnx_asr.preprocessors import Preprocessor
from onnx_asr.utils import pad_list
from preprocessors import nemo


@pytest.fixture(scope="module")
def preprocessor_origin(request):
    preprocessor = AudioToMelSpectrogramPreprocessor(
        window_size=nemo.win_length / nemo.sample_rate,
        window_stride=nemo.hop_length / nemo.sample_rate,
        features=request.param,
        n_fft=nemo.n_fft,
        pad_to=0,
    )
    preprocessor.eval()
    return preprocessor


def preprocessor_torch(waveforms, lens, n_mels):
    waveforms = torch.from_numpy(waveforms)
    if nemo.preemph != 0.0:
        timemask = torch.arange(waveforms.shape[-1]).unsqueeze(0) < torch.from_numpy(lens).unsqueeze(1)
        waveforms = torch.cat((waveforms[:, :1], waveforms[:, 1:] - nemo.preemph * waveforms[:, :-1]), dim=1)
        waveforms = waveforms.masked_fill(~timemask, 0.0)

    spectrogram = torchaudio.functional.spectrogram(
        waveforms,
        pad=0,
        window=torch.hann_window(nemo.win_length, periodic=False),
        n_fft=nemo.n_fft,
        hop_length=nemo.hop_length,
        win_length=nemo.win_length,
        power=2,
        normalized=False,
        pad_mode="constant",
    )
    mel_spectrogram = torch.matmul(
        spectrogram.transpose(-1, -2), torch.from_numpy(nemo.melscale_fbanks80 if n_mels == 80 else nemo.melscale_fbanks128)
    ).transpose(-1, -2)
    log_mel_spectrogram = torch.log(mel_spectrogram + nemo.log_zero_guard_value)

    features_lens = torch.from_numpy(lens) // nemo.hop_length
    mask = torch.arange(log_mel_spectrogram.shape[-1]) < features_lens[:, None, None]
    mean = torch.where(mask, log_mel_spectrogram, 0).sum(dim=-1, keepdim=True) / features_lens[:, None, None]
    var = torch.where(mask, (log_mel_spectrogram - mean) ** 2, 0).sum(dim=-1, keepdim=True) / (features_lens[:, None, None] - 1)
    features = torch.where(mask, (log_mel_spectrogram - mean) / (var.sqrt() + 1e-5), 0).numpy()
    return features, features_lens.numpy()


def preprocessor_torch80(waveforms, lens):
    return preprocessor_torch(waveforms, lens, 80)


def preprocessor_torch128(waveforms, lens):
    return preprocessor_torch(waveforms, lens, 128)


@pytest.fixture(scope="module")
def preprocessor(request):
    match request.param:
        case "torch 80":
            return preprocessor_torch80
        case "torch 128":
            return preprocessor_torch128
        case "onnx_func 80":
            return nemo.NemoPreprocessor80
        case "onnx_func 128":
            return nemo.NemoPreprocessor128
        case "onnx_model 80":
            return Preprocessor("nemo80", {})
        case "onnx_model 128":
            return Preprocessor("nemo128", {})
        case "onnx_model_mt 80":
            return Preprocessor("nemo80", {"max_concurrent_workers": 2})
        case "onnx_model_mt 128":
            return Preprocessor("nemo128", {"max_concurrent_workers": 2})


@pytest.mark.parametrize(
    ("preprocessor_origin", "preprocessor"),
    [
        (80, "torch 80"),
        (128, "torch 128"),
        (80, "onnx_func 80"),
        (128, "onnx_func 128"),
        (80, "onnx_model 80"),
        (128, "onnx_model 128"),
        (80, "onnx_model_mt 80"),
        (128, "onnx_model_mt 128"),
    ],
    indirect=["preprocessor_origin", "preprocessor"],
)
def test_nemo_preprocessor(preprocessor_origin, preprocessor, waveforms):
    waveforms, lens = pad_list(waveforms)
    expected, expected_lens = preprocessor_origin(input_signal=torch.from_numpy(waveforms), length=torch.from_numpy(lens))
    actual, actual_lens = preprocessor(waveforms, lens)

    assert expected.shape[2] >= max(expected_lens)
    np.testing.assert_equal(actual_lens, expected_lens.numpy())
    np.testing.assert_allclose(actual, expected.numpy(), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    ("preprocessor_origin", "melscale_fbanks"),
    [(80, nemo.melscale_fbanks80), (128, nemo.melscale_fbanks128)],
    indirect=["preprocessor_origin"],
)
def test_nemo_melscale_fbanks(preprocessor_origin, melscale_fbanks):
    expected = preprocessor_origin.filter_banks[0].T.numpy()

    np.testing.assert_allclose(melscale_fbanks, expected, atol=5e-7)
