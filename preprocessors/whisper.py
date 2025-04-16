import torch
import torchaudio
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import FLOAT, INT64, script
from onnxscript import opset17 as op
from whisper.audio import log_mel_spectrogram, mel_filters

sample_rate = 16_000
n_fft = 400
win_length = 400
hop_length = 160
n_mels = 80

clamp_min = 1e-10
ln10 = 2.302585092994046

melscale_fbanks = mel_filters("cpu", n_mels).T


def whisper_preprocessor_origin(waveforms, lens):
    return log_mel_spectrogram(waveforms, n_mels).numpy(), lens // hop_length


def whisper_preprocessor_torch(waveforms, lens):
    waveforms = torch.from_numpy(waveforms)
    spectrogram = torchaudio.functional.spectrogram(
        waveforms,
        pad=0,
        window=torch.hann_window(win_length),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=2,
        normalized=False,
    )[..., :-1]
    mel_spectrogram = torch.matmul(spectrogram.transpose(-1, -2), melscale_fbanks).transpose(-1, -2)
    log_mel_spectrogram = torch.clamp(mel_spectrogram, min=clamp_min).log10()
    features = (torch.maximum(log_mel_spectrogram, log_mel_spectrogram.max() - 8.0) + 4.0) / 4.0
    return features, lens // hop_length


@script(doc_string="LogMelSpectrogram feature extractor for Whisper models")
def WhisperPreprocessor(
    waveforms: FLOAT["batch_size", "N"],  # noqa: F821
    waveforms_lens: INT64["batch_size"],  # noqa: F821
) -> tuple[FLOAT["batch_size", n_mels, "T"], INT64["batch_size"]]:  # noqa: F821
    waveforms = op.Pad(
        waveforms,
        pads=op.Constant(value=make_tensor("pads", TensorProto.INT64, (4,), [0, n_fft // 2, 0, n_fft // 2])),
        mode="reflect",
    )

    hann_window = op.HannWindow(win_length, output_datatype=TensorProto.DOUBLE)
    image = op.STFT(op.CastLike(waveforms, hann_window), hop_length, hann_window)[:, :-1]
    spectrogram = op.ReduceSumSquare(image, axes=[-1], keepdims=0)

    melscale_fbanks_tensor = op.Constant(
        value=make_tensor("melscale_fbanks", TensorProto.FLOAT, melscale_fbanks.shape, melscale_fbanks.numpy())
    )
    mel_spectrogram = op.MatMul(op.CastLike(spectrogram, melscale_fbanks_tensor), melscale_fbanks_tensor)
    log_mel_spectrogram = op.Log(op.Clip(mel_spectrogram, clamp_min)) / ln10
    log_mel_spectrogram = (op.Max(log_mel_spectrogram, op.ReduceMax(log_mel_spectrogram) - 8) + 4) / 4.0

    features_lens = waveforms_lens / hop_length
    features = op.Transpose(log_mel_spectrogram, perm=[0, 2, 1])
    return features, features_lens
