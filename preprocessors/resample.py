import math
from typing import Sequence  # noqa: UP035

import torch
import torchaudio
from onnx import numpy_helper
from onnxscript import FLOAT, INT64, script
from onnxscript import opset17 as op


def make_kernel(orig_freq: int):
    new_freq = 16_000
    gcd = math.gcd(orig_freq, new_freq)
    kernel, width = torchaudio.functional.functional._get_sinc_resample_kernel(orig_freq, new_freq, gcd, dtype=torch.float32)
    return numpy_helper.from_array(kernel.numpy()[:, None], "kernel"), width, orig_freq // gcd, new_freq // gcd


kernel08, width08, orig_freq08, new_freq08 = make_kernel(8_000)
kernel22, width22, orig_freq22, new_freq22 = make_kernel(22_050)
kernel24, width24, orig_freq24, new_freq24 = make_kernel(24_000)
kernel32, width32, orig_freq32, new_freq32 = make_kernel(32_000)
kernel44, width44, orig_freq44, new_freq44 = make_kernel(44_100)
kernel48, width48, orig_freq48, new_freq48 = make_kernel(48_000)


@script()
def resample(
    waveforms: FLOAT["batch_size", "N"],
    waveforms_lens: INT64["batch_size"],
    kernel: FLOAT["k1", 1, 1, "k2"],
    pads: Sequence[int],
    strides: Sequence[int],
    orig_freq: int,
    new_freq: int,
) -> tuple[FLOAT["batch_size", "M"], INT64["batch_size"]]:
    conv = op.Conv(op.Unsqueeze(waveforms, axes=[1, 2]), kernel, pads=pads, strides=strides)
    resampled = op.Flatten(op.Transpose(conv, perm=(0, 3, 2, 1)))
    resampled_lens = (new_freq * waveforms_lens + orig_freq - 1) / orig_freq

    new_len = (new_freq * op.Shape(waveforms, start=1, end=2)[0] + orig_freq - 1) / orig_freq
    mask = op.Unsqueeze(op.Range(0, new_len, 1), [0]) < op.Unsqueeze(resampled_lens, [1])
    return op.Where(mask, resampled[:, :new_len], 0), resampled_lens


@script(doc_string="Resampling waveform to 16 kHz")
def ResamplePreprocessor(
    waveforms: FLOAT["batch_size", "N"],
    waveforms_lens: INT64["batch_size"],
    sample_rate: INT64[1],
) -> tuple[FLOAT["batch_size", "M"], INT64["batch_size"]]:
    if sample_rate[0] == 8_000:
        waveforms, waveforms_lens = resample(
            waveforms,
            waveforms_lens,
            op.Constant(value=kernel08),
            (0, width08, 0, width08 + orig_freq08),
            (1, orig_freq08),
            orig_freq08,
            new_freq08,
        )
    elif sample_rate[0] == 22_050:
        waveforms, waveforms_lens = resample(
            waveforms,
            waveforms_lens,
            op.Constant(value=kernel22),
            (0, width22, 0, width22 + orig_freq22),
            (1, orig_freq22),
            orig_freq22,
            new_freq22,
        )
    elif sample_rate[0] == 24_000:
        waveforms, waveforms_lens = resample(
            waveforms,
            waveforms_lens,
            op.Constant(value=kernel24),
            (0, width24, 0, width24 + orig_freq24),
            (1, orig_freq24),
            orig_freq24,
            new_freq24,
        )
    elif sample_rate[0] == 32_000:
        waveforms, waveforms_lens = resample(
            waveforms,
            waveforms_lens,
            op.Constant(value=kernel32),
            (0, width32, 0, width32 + orig_freq32),
            (1, orig_freq32),
            orig_freq32,
            new_freq32,
        )
    elif sample_rate[0] == 44_100:
        waveforms, waveforms_lens = resample(
            waveforms,
            waveforms_lens,
            op.Constant(value=kernel44),
            (0, width44, 0, width44 + orig_freq44),
            (1, orig_freq44),
            orig_freq44,
            new_freq44,
        )
    elif sample_rate[0] == 48_000:
        waveforms, waveforms_lens = resample(
            waveforms,
            waveforms_lens,
            op.Constant(value=kernel48),
            (0, width48, 0, width48 + orig_freq48),
            (1, orig_freq48),
            orig_freq48,
            new_freq48,
        )

    resampled = op.Identity(waveforms)
    resampled_lens = op.Identity(waveforms_lens)
    return resampled, resampled_lens
