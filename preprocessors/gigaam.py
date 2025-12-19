import torch
import torchaudio
from onnxscript import DOUBLE, FLOAT, INT64, script
from onnxscript import opset17 as op

sample_rate = 16_000
n_fft_v2 = sample_rate // 40
n_fft_v3 = sample_rate // 50
win_length_v2 = sample_rate // 40
win_length_v3 = sample_rate // 50
hop_length = sample_rate // 100
n_mels = 64

f_min = 0
f_max = 8_000

clamp_min = 1e-9
clamp_max = 1e9

melscale_fbanks_v2 = torchaudio.functional.melscale_fbanks(n_fft_v2 // 2 + 1, f_min, f_max, n_mels, sample_rate).numpy()
melscale_fbanks_v3 = (
    torchaudio.functional.melscale_fbanks(n_fft_v3 // 2 + 1, f_min, f_max, n_mels, sample_rate).bfloat16().float().numpy()
)
hann_window_v3 = torch.hann_window(win_length_v3).bfloat16().double().numpy()


@script(doc_string="LogMelSpectrogram feature extractor for GigaAM v2 models")
def GigaamPreprocessorV2(
    waveforms: FLOAT["batch_size", "N"],
    waveforms_lens: INT64["batch_size"],
) -> tuple[FLOAT["batch_size", n_mels, "T"], INT64["batch_size"]]:
    waveforms = op.Pad(
        waveforms,
        pads=op.Constant(value=[0, n_fft_v2 // 2, 0, n_fft_v2 // 2]),
        mode="reflect",
    )

    hann_window = op.HannWindow(win_length_v2, output_datatype=DOUBLE.dtype)
    image = op.STFT(op.CastLike(waveforms, hann_window), hop_length, hann_window)
    spectrogram = op.ReduceSumSquare(image, axes=[-1], keepdims=0)

    mel_spectrogram = op.MatMul(op.CastLike(spectrogram, melscale_fbanks_v2), melscale_fbanks_v2)
    log_mel_spectrogram = op.Log(op.Clip(mel_spectrogram, clamp_min, clamp_max))

    features_lens = waveforms_lens / hop_length + 1
    features = op.Transpose(log_mel_spectrogram, perm=[0, 2, 1])
    return features, features_lens


@script(doc_string="LogMelSpectrogram feature extractor for GigaAM v3 models")
def GigaamPreprocessorV3(
    waveforms: FLOAT["batch_size", "N"],
    waveforms_lens: INT64["batch_size"],
) -> tuple[FLOAT["batch_size", n_mels, "T"], INT64["batch_size"]]:
    image = op.STFT(op.CastLike(waveforms, hann_window_v3), hop_length, hann_window_v3)
    spectrogram = op.ReduceSumSquare(image, axes=[-1], keepdims=0)

    mel_spectrogram = op.MatMul(op.CastLike(spectrogram, melscale_fbanks_v3), melscale_fbanks_v3)
    log_mel_spectrogram = op.Log(op.Clip(mel_spectrogram, clamp_min, clamp_max))

    features_lens = (waveforms_lens - win_length_v3) / hop_length + 1
    features = op.Transpose(log_mel_spectrogram, perm=[0, 2, 1])
    return features, features_lens
