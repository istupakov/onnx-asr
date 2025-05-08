import torchaudio
from onnx import TensorProto, numpy_helper
from onnxscript import FLOAT, INT64, script
from onnxscript import opset17 as op

sample_rate = 16_000
n_fft = 512
win_length = 400
hop_length = 160
preemph = 0.97

log_zero_guard_value = float(2**-24)

melscale_fbanks80 = torchaudio.functional.melscale_fbanks(
    n_fft // 2 + 1, 0, sample_rate // 2, 80, sample_rate, "slaney", "slaney"
)
melscale_fbanks128 = torchaudio.functional.melscale_fbanks(
    n_fft // 2 + 1, 0, sample_rate // 2, 128, sample_rate, "slaney", "slaney"
)


@script()
def normalize(x: FLOAT["B", "M", "T"], lens: INT64["B"]):
    lens_3d = op.Unsqueeze(lens, [1, 2])
    mask = op.Range(0, op.Shape(x)[-1], 1) < lens_3d
    lens_3d = op.CastLike(lens_3d, x)
    mean = op.ReduceSum(op.Where(mask, x, 0), axes=[-1], keepdims=1) / lens_3d
    var = op.ReduceSumSquare(op.Where(mask, x - mean, 0), axes=(-1,), keepdims=1) / (lens_3d - 1)
    return op.Where(mask, (x - mean) / (op.Sqrt(var) + 1e-5), 0)


@script()
def nemo_preprocessor(waveforms: FLOAT["B", "N"], waveforms_lens: INT64["B"], melscale_fbanks: FLOAT[n_fft // 2 + 1, "M"]):
    if preemph != 0.0:
        waveforms = op.Concat(waveforms[:, :1], waveforms[:, 1:] - preemph * waveforms[:, :-1], axis=-1)

    waveforms = op.Pad(
        waveforms,
        pads=op.Constant(value_ints=(0, n_fft // 2, 0, n_fft // 2)),
        mode="reflect",
    )
    hann_window = op.Pad(
        op.HannWindow(win_length, periodic=0, output_datatype=TensorProto.DOUBLE),
        pads=op.Constant(value_ints=(n_fft // 2 - win_length // 2, n_fft // 2 - win_length // 2)),
    )
    image = op.STFT(op.CastLike(waveforms, hann_window), hop_length, hann_window)
    spectrogram = op.ReduceSumSquare(image, axes=(-1,), keepdims=0)

    mel_spectrogram = op.MatMul(op.CastLike(spectrogram, melscale_fbanks), melscale_fbanks)
    log_mel_spectrogram = op.Log(mel_spectrogram + log_zero_guard_value)

    features_lens = waveforms_lens / hop_length + 1
    return normalize(op.Transpose(log_mel_spectrogram, perm=(0, 2, 1)), features_lens), features_lens


@script(doc_string="LogMelSpectrogram feature extractor for Nemo models")
def NemoPreprocessor80(
    waveforms: FLOAT["batch_size", "N"],
    waveforms_lens: INT64["batch_size"],
) -> tuple[FLOAT["batch_size", 80, "T"], INT64["batch_size"]]:
    features, features_lens = nemo_preprocessor(
        waveforms,
        waveforms_lens,
        op.Constant(value=numpy_helper.from_array(melscale_fbanks80.numpy(), "melscale_fbanks")),
    )
    return features, features_lens


@script(doc_string="LogMelSpectrogram feature extractor for Nemo models")
def NemoPreprocessor128(
    waveforms: FLOAT["batch_size", "N"],
    waveforms_lens: INT64["batch_size"],
) -> tuple[FLOAT["batch_size", 128, "T"], INT64["batch_size"]]:
    features, features_lens = nemo_preprocessor(
        waveforms,
        waveforms_lens,
        op.Constant(value=numpy_helper.from_array(melscale_fbanks128.numpy(), "melscale_fbanks")),
    )
    return features, features_lens
