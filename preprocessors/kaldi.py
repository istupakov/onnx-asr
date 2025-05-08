import numpy as np
import torch
import torchaudio
from onnx import TensorProto, numpy_helper
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
mel_banks = torch.nn.functional.pad(mel_banks, (0, 1)).T


@script()
def symmetric_pad(waveforms: FLOAT["B", "N"], lens: INT64["B"]):
    @graph()
    def pad(waveform: FLOAT["N"], len: INT64):
        pad_left = op.Constant(value_int=win_length // 2 - hop_length // 2)
        pad_right = op.Constant(value_int=win_length // 2)

        return op.Concat(
            waveform[pad_left - 1 :: -1], waveform[:len], waveform[len - 1 : len - pad_right - 1 : -1], waveform[len:], axis=-1
        )

    return op.Cast(op.Scan(waveforms, lens, body=pad, num_scan_inputs=2), to=TensorProto.FLOAT)


@script()
def sliding_window(waveform: FLOAT["B", "N"]):
    samples = op.Shape(waveform)[-1]
    waveform = waveform[:, : samples - (samples + hop_length - win_length) % hop_length]

    X0 = waveform[:, : win_length - hop_length]
    X = op.Reshape(
        waveform[:, win_length - hop_length :],
        shape=op.Constant(value_ints=(0, -1, hop_length)),
    )

    @graph()
    def sliding_buffer(prev: FLOAT["B", win_length - hop_length], curr: FLOAT["B", hop_length]):
        hop_len = op.Constant(value_int=hop_length // 1)
        frame = op.Concat(prev, curr, axis=-1)
        next = frame[:, hop_len:]
        return next, frame

    _, frames = op.Scan(X0, X, body=sliding_buffer, num_scan_inputs=1, scan_input_axes=(1,), scan_output_axes=(1,))
    return op.Cast(frames, to=TensorProto.FLOAT)


@script()
def normalize(frames: FLOAT["B", "T", win_length]):
    if dither != 0.0:
        frames = frames + op.RandomNormalLike(frames, scale=dither)

    if remove_dc_offset:
        mean = op.ReduceMean(frames, axes=(-1,))
        frames = frames - mean

    if preemphasis_coefficient != 0.0:
        offset = op.Pad(frames, pads=[0, 0, 1, 0, 0, -1], mode="edge")
        frames = frames - preemphasis_coefficient * offset

    return frames


@script(doc_string="LogMelSpectrogram feature extractor for Kaldi models")
def KaldiPreprocessor(
    waveforms: FLOAT["batch_size", "N"],
    waveforms_lens: INT64["batch_size"],
) -> tuple[FLOAT["batch_size", "T", num_mel_bins], INT64["batch_size"]]:
    waveforms = symmetric_pad(waveforms, waveforms_lens)
    frames = sliding_window(waveforms)
    frames = normalize(frames)

    povey_window = op.Pow(op.HannWindow(win_length, periodic=0), 0.85)
    frames = povey_window * frames

    image = op.DFT(op.Unsqueeze(frames, axes=-1), n_fft, axis=-2, onesided=1)
    spectrogram = op.ReduceSumSquare(image, axes=(-1,), keepdims=0)

    mel_banks_tensor = op.Constant(value=numpy_helper.from_array(mel_banks.numpy(), "mel_banks"))
    mel_spectrogram = op.MatMul(spectrogram, mel_banks_tensor)
    log_mel_spectrogram = op.Log(op.Clip(mel_spectrogram, min=float_eps))

    features_lens = (waveforms_lens + hop_length / 2) / hop_length
    mask = op.Unsqueeze(op.Range(0, op.Shape(log_mel_spectrogram)[1], 1), [0, 2]) < op.Unsqueeze(features_lens, [1, 2])
    features = op.Where(mask, log_mel_spectrogram, 0)
    return features, features_lens
