import numpy as np
import pytest
import torch
import torchaudio
from whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, mel_filters, pad_or_trim

from onnx_asr.preprocessors import Preprocessor
from onnx_asr.utils import pad_list
from preprocessors import whisper


def preprocessor_origin(waveforms, lens, n_mels):
    waveforms = pad_or_trim(waveforms, N_SAMPLES)
    return log_mel_spectrogram(waveforms, n_mels).numpy(), np.full_like(lens, N_FRAMES)


def preprocessor_torch(waveforms, lens, n_mels):
    waveforms = torch.from_numpy(waveforms)
    waveforms = waveforms[:, : whisper.chunk_length * whisper.sample_rate]
    waveforms = torch.nn.functional.pad(waveforms, (0, whisper.chunk_length * whisper.sample_rate - waveforms.shape[-1]))
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
        spectrogram.transpose(-1, -2), torch.from_numpy(whisper.melscale_fbanks80 if n_mels == 80 else whisper.melscale_fbanks128)
    ).transpose(-1, -2)
    log_mel_spectrogram = torch.clamp(mel_spectrogram, min=whisper.clamp_min).log10()
    features = (torch.maximum(log_mel_spectrogram, log_mel_spectrogram.max() - 8.0) + 4.0) / 4.0
    return features, np.full_like(lens, whisper.chunk_length * whisper.sample_rate // whisper.hop_length)


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
            return whisper.WhisperPreprocessor80
        case "onnx_func 128":
            return whisper.WhisperPreprocessor128
        case "onnx_model 80":
            return Preprocessor("whisper80", {})
        case "onnx_model 128":
            return Preprocessor("whisper128", {})


@pytest.mark.parametrize(
    ("n_mels", "preprocessor"),
    [
        (80, "torch 80"),
        (128, "torch 128"),
        (80, "onnx_func 80"),
        (128, "onnx_func 128"),
        (80, "onnx_model 80"),
        (128, "onnx_model 128"),
    ],
    indirect=["preprocessor"],
)
def test_whisper_preprocessor(n_mels, preprocessor, waveforms):
    waveforms, lens = pad_list(waveforms)
    expected, expected_lens = preprocessor_origin(waveforms, lens, n_mels)
    actual, actual_lens = preprocessor(waveforms, lens)

    assert expected.shape[2] == max(expected_lens)
    np.testing.assert_equal(actual_lens, expected_lens)
    np.testing.assert_allclose(actual, expected, atol=5e-5)


@pytest.mark.parametrize(("n_mels", "melscale_fbanks"), [(80, whisper.melscale_fbanks80), (128, whisper.melscale_fbanks128)])
def test_whisper_melscale_fbanks(n_mels, melscale_fbanks):
    expected = mel_filters("cpu", n_mels).T.numpy()

    np.testing.assert_allclose(melscale_fbanks, expected, atol=5e-7)
