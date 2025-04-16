import torch
import torchaudio
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import FLOAT, INT64, script
from onnxscript import opset17 as op
from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor

sample_rate = 16_000
n_fft = 512
win_length = 400
hop_length = 160
n_mels = 80
preemph = 0.97

log_zero_guard_value = 2**-24

preprocessor = AudioToMelSpectrogramPreprocessor(
    window_size=win_length / sample_rate,
    window_stride=hop_length / sample_rate,
    features=n_mels,
    n_fft=n_fft,
    pad_to=0,
)
melscale_fbanks = preprocessor.filter_banks[0].T


def nemo_preprocessor_origin(waveforms, lens):
    preprocessor.eval()
    features, out_lens = preprocessor(input_signal=torch.from_numpy(waveforms), length=torch.from_numpy(lens))
    return features.numpy(), out_lens.numpy()


def nemo_preprocessor_torch(waveforms, lens):
    waveforms = torch.from_numpy(waveforms)
    if preemph != 0.0:
        waveforms = torch.cat((waveforms[:, :1], waveforms[:, 1:] - preemph * waveforms[:, :-1]), dim=1)
    spectrogram = torchaudio.functional.spectrogram(
        waveforms,
        pad=0,
        window=torch.hann_window(win_length, periodic=False),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=2,
        normalized=False,
    )
    mel_spectrogram = torch.matmul(spectrogram.transpose(-1, -2), melscale_fbanks).transpose(-1, -2)
    log_mel_spectrogram = torch.log(mel_spectrogram + log_zero_guard_value)

    features_lens = torch.from_numpy(lens) // hop_length + 1
    mask = torch.arange(log_mel_spectrogram.shape[-1]) < features_lens[:, None, None]
    mean = torch.where(mask, log_mel_spectrogram, 0).sum(dim=-1, keepdim=True) / features_lens[:, None, None]
    var = torch.where(mask, (log_mel_spectrogram - mean) ** 2, 0).sum(dim=-1, keepdim=True) / (features_lens[:, None, None] - 1)
    features = torch.where(mask, (log_mel_spectrogram - mean) / (var.sqrt() + 1e-5), 0).numpy()
    return features, features_lens.numpy()


@script()
def normalize(x, lens):
    lens_3d = op.Unsqueeze(lens, [1, 2])
    mask = op.Range(0, op.Shape(x)[-1], 1) < lens_3d
    lens_3d = op.CastLike(lens_3d, x)
    mean = op.ReduceSum(op.Where(mask, x, 0), axes=[-1], keepdims=1) / lens_3d
    var = op.ReduceSumSquare(op.Where(mask, x - mean, 0), axes=[-1], keepdims=1) / (lens_3d - 1)
    return op.Where(mask, (x - mean) / (op.Sqrt(var) + 1e-5), 0)


@script(doc_string="LogMelSpectrogram feature extractor for Nemo models")
def NemoPreprocessor(
    waveforms: FLOAT["batch_size", "N"],  # noqa: F821
    waveforms_lens: INT64["batch_size"],  # noqa: F821
) -> tuple[FLOAT["batch_size", n_mels, "T"], INT64["batch_size"]]:  # noqa: F821
    if preemph != 0.0:
        waveforms = op.Concat(waveforms[:, :1], waveforms[:, 1:] - preemph * waveforms[:, :-1], axis=-1)

    waveforms = op.Pad(
        waveforms,
        pads=op.Constant(value=make_tensor("waveform_pads", TensorProto.INT64, (4,), [0, n_fft // 2, 0, n_fft // 2])),
        mode="reflect",
    )
    hann_window = op.Pad(
        op.HannWindow(win_length, periodic=0, output_datatype=TensorProto.DOUBLE),
        pads=op.Constant(
            value=make_tensor(
                "window_pads", TensorProto.INT64, (2,), [n_fft // 2 - win_length // 2, n_fft // 2 - win_length // 2]
            )
        ),
    )
    image = op.STFT(op.CastLike(waveforms, hann_window), hop_length, hann_window)
    spectrogram = op.ReduceSumSquare(image, axes=[-1], keepdims=0)

    melscale_fbanks_tensor = op.Constant(
        value=make_tensor("melscale_fbanks", TensorProto.FLOAT, melscale_fbanks.shape, melscale_fbanks.numpy())
    )
    mel_spectrogram = op.MatMul(op.CastLike(spectrogram, melscale_fbanks_tensor), melscale_fbanks_tensor)
    log_mel_spectrogram = op.Log(mel_spectrogram + log_zero_guard_value)

    features_lens = waveforms_lens / hop_length + 1
    features = normalize(op.Transpose(log_mel_spectrogram, perm=[0, 2, 1]), features_lens)
    return features, features_lens
