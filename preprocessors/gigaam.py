import torch
import torchaudio
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import FLOAT, INT64, script
from onnxscript import opset17 as op

sample_rate = 16_000
n_fft = sample_rate // 40
win_length = sample_rate // 40
hop_length = sample_rate // 100
n_mels = 64

f_min = 0
f_max = 8_000

clamp_min = 1e-9
clamp_max = 1e9

melscale_fbanks = torchaudio.functional.melscale_fbanks(n_fft // 2 + 1, f_min, f_max, n_mels, sample_rate)


def gigaam_preprocessor_origin(waveforms, lens):
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    features_lens = torch.from_numpy(lens).div(hop_length, rounding_mode="floor").add(1).long().numpy()
    return torch.log(transform(torch.from_numpy(waveforms)).clamp_(clamp_min, clamp_max)).numpy(), features_lens


def gigaam_preprocessor_torch(waveforms, lens):
    waveforms = torch.from_numpy(waveforms)
    spectrogram = torchaudio.functional.spectrogram(
        waveforms,
        pad=0,
        window=torch.hann_window(win_length),
        n_fft=win_length,
        hop_length=hop_length,
        win_length=win_length,
        power=2,
        normalized=False,
    )
    mel_spectrogram = torch.matmul(spectrogram.transpose(-1, -2), melscale_fbanks).transpose(-1, -2)
    return torch.log(mel_spectrogram.clamp_(clamp_min, clamp_max)).numpy(), lens // hop_length + 1


@script(doc_string="LogMelSpectrogram feature extractor for GigaAM models")
def GigaamPreprocessor(
    waveforms: FLOAT["batch_size", "N"],  # noqa: F821
    waveforms_lens: INT64["batch_size"],  # noqa: F821
) -> tuple[FLOAT["batch_size", n_mels, "T"], INT64["batch_size"]]:  # noqa: F821
    waveforms = op.Pad(
        waveforms,
        pads=op.Constant(value=make_tensor("pads", TensorProto.INT64, (4,), [0, n_fft // 2, 0, n_fft // 2])),
        mode="reflect",
    )

    hann_window = op.HannWindow(win_length, output_datatype=TensorProto.DOUBLE)
    image = op.STFT(op.CastLike(waveforms, hann_window), hop_length, hann_window)
    spectrogram = op.ReduceSumSquare(image, axes=[-1], keepdims=0)

    melscale_fbanks_tensor = op.Constant(
        value=make_tensor("melscale_fbanks", TensorProto.FLOAT, melscale_fbanks.shape, melscale_fbanks.numpy())
    )
    mel_spectrogram = op.MatMul(op.CastLike(spectrogram, melscale_fbanks_tensor), melscale_fbanks_tensor)
    log_mel_spectrogram = op.Log(op.Clip(mel_spectrogram, clamp_min, clamp_max))

    features_lens = waveforms_lens / hop_length + 1
    features = op.Transpose(log_mel_spectrogram, perm=[0, 2, 1])
    return features, features_lens
