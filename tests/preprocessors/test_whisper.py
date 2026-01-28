import numpy as np
import pytest
import torch
import torchaudio
from whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, mel_filters, pad_or_trim

from onnx_asr.preprocessors.preprocessor import OnnxPreprocessor
from onnx_asr.utils import pad_list
from preprocessors import whisper


@pytest.fixture
def base_sec():
    return 30


@pytest.fixture(scope="module", params=[80, 128])
def n_mels(request):
    return request.param


def preprocessor_origin(waveforms, lens, n_mels):
    waveforms = pad_or_trim(waveforms, N_SAMPLES)
    return log_mel_spectrogram(waveforms, n_mels).numpy(), np.full_like(lens, N_FRAMES)


def preprocessor_torch(waveforms, lens, n_mels):
    waveforms = torch.from_numpy(waveforms)
    waveforms = waveforms[:, : whisper.chunk_length * whisper.sample_rate]
    waveforms = torch.nn.functional.pad(
        waveforms, (0, whisper.chunk_length * whisper.sample_rate - waveforms.shape[-1])
    )
    spectrogram = torchaudio.functional.spectrogram(
        waveforms,
        pad=0,
        window=torch.hann_window(whisper.win_length),
        n_fft=whisper.n_fft,
        hop_length=whisper.hop_length,
        win_length=whisper.win_length,
        power=2,
        normalized=False,
    )[..., :-1]
    mel_spectrogram = torch.matmul(
        spectrogram.transpose(-1, -2),
        torch.from_numpy(whisper.melscale_fbanks80 if n_mels == 80 else whisper.melscale_fbanks128),
    ).transpose(-1, -2)
    log_mel_spectrogram = torch.clamp(mel_spectrogram, min=whisper.clamp_min).log10()
    features = (torch.maximum(log_mel_spectrogram, log_mel_spectrogram.max() - 8.0) + 4.0) / 4.0
    return features, np.full_like(lens, whisper.chunk_length * whisper.sample_rate // whisper.hop_length)


@pytest.fixture(scope="module", params=["torch", "onnx_func", "onnx_model"])
def preprocessor(request, n_mels):
    match request.param:
        case "torch":
            return lambda waveforms, lens: preprocessor_torch(waveforms, lens, n_mels)
        case "onnx_func":
            return whisper.WhisperPreprocessor80 if n_mels == 80 else whisper.WhisperPreprocessor128
        case "onnx_model":
            return OnnxPreprocessor(f"whisper{n_mels}", {})


def test_whisper_preprocessor(n_mels, preprocessor, waveforms):
    waveforms, lens = pad_list(waveforms)
    expected, expected_lens = preprocessor_origin(waveforms, lens, n_mels)
    actual, actual_lens = preprocessor(waveforms, lens)

    assert expected.shape[2] == max(expected_lens)
    np.testing.assert_equal(actual_lens, expected_lens)
    np.testing.assert_allclose(actual, expected, atol=5e-5)


def test_whisper_melscale_fbanks(n_mels):
    expected = mel_filters("cpu", n_mels).T.numpy()
    melscale_fbanks = whisper.melscale_fbanks80 if n_mels == 80 else whisper.melscale_fbanks128

    np.testing.assert_allclose(melscale_fbanks, expected, atol=5e-7)
