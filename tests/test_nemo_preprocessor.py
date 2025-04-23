import numpy as np
import pytest
import torch
import torchaudio
from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor

from onnx_asr.preprocessors import Preprocessor
from onnx_asr.utils import pad_list
from preprocessors import nemo


@pytest.fixture(scope="module")
def preprocessor_origin():
    preprocessor = AudioToMelSpectrogramPreprocessor(
        window_size=nemo.win_length / nemo.sample_rate,
        window_stride=nemo.hop_length / nemo.sample_rate,
        features=nemo.n_mels,
        n_fft=nemo.n_fft,
        pad_to=0,
    )
    preprocessor.eval()
    return preprocessor


def preprocessor_torch(waveforms, lens):
    waveforms = torch.from_numpy(waveforms)
    if nemo.preemph != 0.0:
        waveforms = torch.cat((waveforms[:, :1], waveforms[:, 1:] - nemo.preemph * waveforms[:, :-1]), dim=1)

    spectrogram = torchaudio.functional.spectrogram(
        waveforms,
        pad=0,
        window=torch.hann_window(nemo.win_length, periodic=False),
        n_fft=nemo.n_fft,
        hop_length=nemo.hop_length,
        win_length=nemo.win_length,
        power=2,
        normalized=False,
    )
    mel_spectrogram = torch.matmul(spectrogram.transpose(-1, -2), nemo.melscale_fbanks).transpose(-1, -2)
    log_mel_spectrogram = torch.log(mel_spectrogram + nemo.log_zero_guard_value)

    features_lens = torch.from_numpy(lens) // nemo.hop_length + 1
    mask = torch.arange(log_mel_spectrogram.shape[-1]) < features_lens[:, None, None]
    mean = torch.where(mask, log_mel_spectrogram, 0).sum(dim=-1, keepdim=True) / features_lens[:, None, None]
    var = torch.where(mask, (log_mel_spectrogram - mean) ** 2, 0).sum(dim=-1, keepdim=True) / (features_lens[:, None, None] - 1)
    features = torch.where(mask, (log_mel_spectrogram - mean) / (var.sqrt() + 1e-5), 0).numpy()
    return features, features_lens.numpy()


@pytest.fixture(scope="module")
def preprocessor(request):
    match request.param:
        case "torch":
            return preprocessor_torch
        case "onnx_func":
            return nemo.NemoPreprocessor
        case "onnx_model":
            return Preprocessor("nemo")


@pytest.mark.parametrize(
    "preprocessor",
    [
        "torch",
        "onnx_func",
        "onnx_model",
    ],
    indirect=True,
)
def test_nemo_preprocessor(preprocessor_origin, preprocessor, waveforms):
    waveforms, lens = pad_list(waveforms)
    expected, expected_lens = preprocessor_origin(input_signal=torch.from_numpy(waveforms), length=torch.from_numpy(lens))
    actual, actual_lens = preprocessor(waveforms, lens)

    assert expected.shape[2] == max(expected_lens)
    np.testing.assert_equal(actual_lens, expected_lens.numpy())
    np.testing.assert_allclose(actual, expected.numpy(), atol=1e-4)


def test_nemo_melscale_fbanks(preprocessor_origin):
    expected = preprocessor_origin.filter_banks[0].T.numpy()
    actual = nemo.melscale_fbanks.numpy()

    np.testing.assert_allclose(actual, expected, atol=1e-7)
