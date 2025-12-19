import numpy as np
import torch
import torchaudio
from onnxscript import FLOAT, INT64, graph, script
from onnxscript import opset17 as op

sample_rate = 16_000
n_fft = 512
win_length = 400
hop_length = 160
num_mel_bins = 80

snip_edges = False
dither = 0.0
remove_dc_offset = True
preemphasis_coefficient = 0.97

low_freq = 20
high_freq = -400

float_eps = float(np.finfo(np.float32).eps)

mel_banks, _ = torchaudio.compliance.kaldi.get_mel_banks(num_mel_bins, n_fft, sample_rate, low_freq, high_freq, 0, 0, 1)
mel_banks = torch.nn.functional.pad(mel_banks, (0, 1)).T.numpy()


@script()
def symmetric_pad(waveforms: FLOAT["batch_size", "N"], waveforms_lens: INT64["batch_size"]):
    pad_left = op.Constant(value=win_length // 2 - hop_length // 2)
    pad_right = op.Constant(value=win_length // 2)

    waveforms = op.Concat(waveforms[:, pad_left - 1 :: -1], waveforms, axis=-1)
    waveforms = op.Pad(waveforms, pads=op.Constant(value=[0, 0, 0, win_length // 2]))

    indices_from = op.Unsqueeze(waveforms_lens, axes=[1]) + pad_left - op.Range(1, pad_right + 1, 1)
    indices_to = op.Unsqueeze(waveforms_lens, axes=[1]) + pad_left + op.Range(0, pad_right, 1)

    return op.ScatterElements(waveforms, indices_to, op.GatherElements(waveforms, indices_from, axis=1), axis=1)


@script()
def sliding_window(waveform: FLOAT["batch_size", "N"]):
    samples = op.Squeeze(op.Shape(waveform, start=1, end=2))
    new_len = samples - (samples - win_length) % hop_length
    X0 = waveform[:, : win_length - hop_length]
    X = op.Reshape(
        op.Slice(
            waveform,
            starts=op.Constant(value=[win_length - hop_length]),
            ends=op.Unsqueeze(new_len, axes=[0]),
            steps=[1],
            axes=[1],
        ),
        shape=op.Constant(value=[0, -1, hop_length]),
    )

    hop_len = op.Constant(value=hop_length)

    @graph()
    def sliding_buffer(prev: FLOAT["batch_size", win_length - hop_length], curr: FLOAT["batch_size", hop_length]):
        frame = op.Concat(prev, curr, axis=-1)
        next = frame[:, hop_len:]
        return next, frame

    _, frames = op.Scan(X0, X, body=sliding_buffer, num_scan_inputs=1, scan_input_axes=[1], scan_output_axes=[1])
    return op.Cast(frames, to=FLOAT.dtype)


@script()
def calc_features(image: FLOAT["batch_size", "T", n_fft // 2 + 1, 2], waveforms_lens: INT64["batch_size"]):
    spectrogram = op.ReduceSumSquare(image, axes=[-1], keepdims=0)

    mel_spectrogram = op.MatMul(spectrogram, mel_banks)
    log_mel_spectrogram = op.Log(op.Clip(mel_spectrogram, min=float_eps))

    if not snip_edges:
        features_lens = (waveforms_lens + hop_length / 2) / hop_length
    else:
        features_lens = 1 + (waveforms_lens - win_length) / hop_length

    mask = op.Unsqueeze(op.Range(0, op.Squeeze(op.Shape(log_mel_spectrogram, start=1, end=2)), 1), axes=[0, 2]) < op.Unsqueeze(
        features_lens, axes=[1, 2]
    )
    return op.Where(mask, log_mel_spectrogram, 0.0), features_lens


@script(doc_string="LogMelSpectrogram feature extractor for Kaldi models")
def KaldiPreprocessor(
    waveforms: FLOAT["batch_size", "N"],
    waveforms_lens: INT64["batch_size"],
) -> tuple[FLOAT["batch_size", "T", num_mel_bins], INT64["batch_size"]]:
    if not snip_edges:
        waveforms = symmetric_pad(waveforms, waveforms_lens)

    frames = sliding_window(waveforms)

    if dither != 0.0:
        frames = frames + op.RandomNormalLike(frames, scale=dither)

    if remove_dc_offset:
        frames = frames - op.ReduceMean(frames, axes=[-1])

    if preemphasis_coefficient != 0.0:
        frames = frames - preemphasis_coefficient * op.Pad(frames, pads=[0, 0, 1, 0, 0, -1], mode="edge")

    povey_window = op.Pow(op.HannWindow(win_length, periodic=0), 0.85)
    frames = povey_window * frames

    image = op.DFT(op.Unsqueeze(frames, axes=[-1]), n_fft, axis=-2, onesided=1)

    features, features_lens = calc_features(image, waveforms_lens)
    return features, features_lens


@script(doc_string="LogMelSpectrogram feature extractor for Kaldi models")
def KaldiPreprocessorFast(
    waveforms: FLOAT["batch_size", "N"],
    waveforms_lens: INT64["batch_size"],
) -> tuple[FLOAT["batch_size", "T", num_mel_bins], INT64["batch_size"]]:
    if dither != 0.0:
        waveforms = waveforms + op.RandomNormalLike(waveforms, scale=dither)

    if remove_dc_offset:
        waveforms = waveforms - op.ReduceMean(waveforms, axes=[-1])

    if not snip_edges:
        waveforms = symmetric_pad(waveforms, waveforms_lens)

    if preemphasis_coefficient != 0.0:
        waveforms = waveforms - preemphasis_coefficient * op.Pad(waveforms, pads=[0, 1, 0, -1], mode="edge")

    waveforms = op.Pad(waveforms, pads=op.Constant(value=[0, 0, 0, n_fft - win_length]))
    povey_window = op.Pad(
        op.Pow(op.HannWindow(win_length, periodic=0), 0.85),
        pads=op.Constant(value=[0, n_fft - win_length]),
    )
    image = op.STFT(op.CastLike(waveforms, povey_window), hop_length, povey_window)

    features, features_lens = calc_features(image, waveforms_lens)
    return features, features_lens
