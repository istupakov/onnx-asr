import numpy as np
import pytest
import torch
import torchaudio
from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor

from onnx_asr.preprocessors.numpy_preprocessor import NemoPreprocessorNumpy
from onnx_asr.preprocessors.preprocessor import ConcurrentPreprocessor, OnnxPreprocessor
from onnx_asr.utils import pad_list
from preprocessors import nemo


@pytest.fixture(scope="module", params=[80, 128])
def n_mels(request):
    return request.param


@pytest.fixture(scope="module")
def preprocessor_origin(n_mels):
    preprocessor = AudioToMelSpectrogramPreprocessor(
        window_size=nemo.win_length / nemo.sample_rate,
        window_stride=nemo.hop_length / nemo.sample_rate,
        features=n_mels,
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
        spectrogram.transpose(-1, -2),
        torch.from_numpy(nemo.melscale_fbanks80 if n_mels == 80 else nemo.melscale_fbanks128),
    ).transpose(-1, -2)
    log_mel_spectrogram = torch.log(mel_spectrogram + nemo.log_zero_guard_value)

    features_lens = torch.from_numpy(lens) // nemo.hop_length
    mask = torch.arange(log_mel_spectrogram.shape[-1]) < features_lens[:, None, None]
    mean = torch.where(mask, log_mel_spectrogram, 0).sum(dim=-1, keepdim=True) / features_lens[:, None, None]
    var = torch.where(mask, (log_mel_spectrogram - mean) ** 2, 0).sum(dim=-1, keepdim=True) / (
        features_lens[:, None, None] - 1
    )
    features = torch.where(mask, (log_mel_spectrogram - mean) / (var.sqrt() + 1e-5), 0).numpy()
    return features, features_lens.numpy()


@pytest.fixture(scope="module", params=["torch", "numpy", "onnx_func", "onnx_model", "onnx_model_mt"])
def preprocessor(request, n_mels):
    match request.param:
        case "torch":
            return lambda waveforms, lens: preprocessor_torch(waveforms, lens, n_mels)
        case "numpy":
            return NemoPreprocessorNumpy(f"nemo{n_mels}")
        case "onnx_func":
            return nemo.NemoPreprocessor80 if n_mels == 80 else nemo.NemoPreprocessor128
        case "onnx_model":
            return OnnxPreprocessor(f"nemo{n_mels}", {})
        case "onnx_model_mt":
            return ConcurrentPreprocessor(OnnxPreprocessor(f"nemo{n_mels}", {}), 2)


def test_nemo_preprocessor(preprocessor_origin, preprocessor, waveforms):
    waveforms, lens = pad_list(waveforms)
    expected, expected_lens = preprocessor_origin(
        input_signal=torch.from_numpy(waveforms), length=torch.from_numpy(lens)
    )
    actual, actual_lens = preprocessor(waveforms, lens)

    assert actual.dtype == np.float32
    assert expected.shape[2] >= max(expected_lens)
    np.testing.assert_equal(actual_lens, expected_lens.numpy())
    np.testing.assert_allclose(actual, expected.numpy(), atol=5e-4, rtol=1e-4)


def test_nemo_melscale_fbanks(preprocessor_origin, n_mels):
    expected = preprocessor_origin.filter_banks[0].T.numpy()
    melscale_fbanks = nemo.melscale_fbanks80 if n_mels == 80 else nemo.melscale_fbanks128

    np.testing.assert_allclose(melscale_fbanks, expected, atol=5e-7)
