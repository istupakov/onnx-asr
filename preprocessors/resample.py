import math

import torch
import torchaudio
from onnx import TensorProto, numpy_helper
from onnxscript import FLOAT, INT64, script
from onnxscript import opset17 as op


def make_kernel(orig_freq: int):
    new_freq = 16_000
    gcd = math.gcd(orig_freq, new_freq)
    kernel, width = torchaudio.functional.functional._get_sinc_resample_kernel(orig_freq, new_freq, gcd, dtype=torch.float32)
    return kernel.numpy()[:, None], width, orig_freq // gcd, new_freq // gcd


kernel08, width08, orig_freq08, new_freq08 = make_kernel(8_000)
kernel22, width22, orig_freq22, new_freq22 = make_kernel(22_050)
kernel44, width44, orig_freq44, new_freq44 = make_kernel(44_100)
kernel48, width48, orig_freq48, new_freq48 = make_kernel(48_000)


@script(doc_string="Resampling waveform to 16 kHz")
def ResamplePreprocessor(
    waveforms: FLOAT["batch_size", "N"],
    waveforms_lens: INT64["batch_size"],
    sample_rate: INT64[1],
) -> tuple[FLOAT["batch_size", "M"], INT64["batch_size"]]:
    waveforms = op.Unsqueeze(waveforms, axes=[1, 2])

    if sample_rate[0] == 8_000:
        kernel = op.Constant(value=numpy_helper.from_array(kernel08, "kernel"))
        conv = op.Conv(waveforms, kernel, pads=(0, width08, 0, width08 + orig_freq08), strides=(1, orig_freq08))
        lens = (new_freq08 * waveforms_lens + orig_freq08 - 1) / orig_freq08
    elif sample_rate[0] == 22_050:
        kernel = op.Constant(value=numpy_helper.from_array(kernel22, "kernel"))
        conv = op.Conv(waveforms, kernel, pads=(0, width22, 0, width22 + orig_freq22), strides=(1, orig_freq22))
        lens = (new_freq22 * waveforms_lens + orig_freq22 - 1) / orig_freq22
    elif sample_rate[0] == 44_100:
        kernel = op.Constant(value=numpy_helper.from_array(kernel44, "kernel"))
        conv = op.Conv(waveforms, kernel, pads=(0, width44, 0, width44 + orig_freq44), strides=(1, orig_freq44))
        lens = (new_freq44 * waveforms_lens + orig_freq44 - 1) / orig_freq44
    elif sample_rate[0] == 48_000:
        kernel = op.Constant(value=numpy_helper.from_array(kernel48, "kernel"))
        conv = op.Conv(waveforms, kernel, pads=(0, width48, 0, width48 + orig_freq48), strides=(1, orig_freq48))
        lens = (new_freq48 * waveforms_lens + orig_freq48 - 1) / orig_freq48
    else:
        conv = waveforms
        lens = waveforms_lens

    conv = op.Cast(conv, to=TensorProto.FLOAT)
    resampled_lens = op.Cast(lens, to=TensorProto.INT64)

    flat = op.Flatten(op.Transpose(conv, perm=(0, 3, 2, 1)))
    max_len = op.ReduceMax(resampled_lens, keepdims=1)

    mask = op.Unsqueeze(op.Range(0, max_len, 1), [0]) < op.Unsqueeze(resampled_lens, [1])
    resampled = op.Where(mask, op.Slice(flat, starts=[0], ends=max_len, axes=[1]), 0)

    return resampled, resampled_lens
