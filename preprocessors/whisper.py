import torchaudio
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import FLOAT, INT64, script
from onnxscript import opset17 as op

sample_rate = 16_000
n_fft = 400
win_length = 400
hop_length = 160

clamp_min = 1e-10
ln10 = 2.302585092994046

melscale_fbanks80 = torchaudio.functional.melscale_fbanks(
    n_fft // 2 + 1, 0, sample_rate // 2, 80, sample_rate, "slaney", "slaney"
)
melscale_fbanks128 = torchaudio.functional.melscale_fbanks(
    n_fft // 2 + 1, 0, sample_rate // 2, 128, sample_rate, "slaney", "slaney"
)


@script()
def whisper_preprocessor(waveforms, waveforms_lens, melscale_fbanks):
    waveforms = op.Pad(
        waveforms,
        pads=op.Constant(value=make_tensor("pads", TensorProto.INT64, (4,), [0, n_fft // 2, 0, n_fft // 2])),
        mode="reflect",
    )

    hann_window = op.HannWindow(win_length, output_datatype=TensorProto.DOUBLE)
    image = op.STFT(op.CastLike(waveforms, hann_window), hop_length, hann_window)[:, :-1]
    spectrogram = op.ReduceSumSquare(image, axes=[-1], keepdims=0)

    mel_spectrogram = op.MatMul(op.CastLike(spectrogram, melscale_fbanks), melscale_fbanks)
    log_mel_spectrogram = op.Log(op.Clip(mel_spectrogram, clamp_min)) / ln10
    log_mel_spectrogram = (op.Max(log_mel_spectrogram, op.ReduceMax(log_mel_spectrogram) - 8) + 4) / 4.0

    return op.Transpose(log_mel_spectrogram, perm=[0, 2, 1]), waveforms_lens / hop_length


@script(doc_string="LogMelSpectrogram feature extractor for Whisper models")
def WhisperPreprocessor80(
    waveforms: FLOAT["batch_size", "N"],
    waveforms_lens: INT64["batch_size"],
) -> tuple[FLOAT["batch_size", 80, "T"], INT64["batch_size"]]:
    features, features_lens = whisper_preprocessor(
        waveforms,
        waveforms_lens,
        op.Constant(value=make_tensor("melscale_fbanks", TensorProto.FLOAT, melscale_fbanks80.shape, melscale_fbanks80.numpy())),
    )
    return features, features_lens


@script(doc_string="LogMelSpectrogram feature extractor for Whisper models")
def WhisperPreprocessor128(
    waveforms: FLOAT["batch_size", "N"],
    waveforms_lens: INT64["batch_size"],
) -> tuple[FLOAT["batch_size", 128, "T"], INT64["batch_size"]]:
    features, features_lens = whisper_preprocessor(
        waveforms,
        waveforms_lens,
        op.Constant(
            value=make_tensor("melscale_fbanks", TensorProto.FLOAT, melscale_fbanks128.shape, melscale_fbanks128.numpy())
        ),
    )
    return features, features_lens
