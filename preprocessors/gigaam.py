import torchaudio
from onnx import TensorProto, numpy_helper
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


@script(doc_string="LogMelSpectrogram feature extractor for GigaAM models")
def GigaamPreprocessor(
    waveforms: FLOAT["batch_size", "N"],
    waveforms_lens: INT64["batch_size"],
) -> tuple[FLOAT["batch_size", n_mels, "T"], INT64["batch_size"]]:
    waveforms = op.Pad(
        waveforms,
        pads=op.Constant(value_ints=(0, n_fft // 2, 0, n_fft // 2)),
        mode="reflect",
    )

    hann_window = op.HannWindow(win_length, output_datatype=TensorProto.DOUBLE)
    image = op.STFT(op.CastLike(waveforms, hann_window), hop_length, hann_window)
    spectrogram = op.ReduceSumSquare(image, axes=(-1,), keepdims=0)

    melscale_fbanks_tensor = op.Constant(value=numpy_helper.from_array(melscale_fbanks.numpy(), "melscale_fbanks"))
    mel_spectrogram = op.MatMul(op.CastLike(spectrogram, melscale_fbanks_tensor), melscale_fbanks_tensor)
    log_mel_spectrogram = op.Log(op.Clip(mel_spectrogram, clamp_min, clamp_max))

    features_lens = waveforms_lens / hop_length + 1
    features = op.Transpose(log_mel_spectrogram, perm=(0, 2, 1))
    return features, features_lens
