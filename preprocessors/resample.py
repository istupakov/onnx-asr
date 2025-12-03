import math
from typing import Sequence  # noqa: UP035

from onnxscript import FLOAT, INT64, script
from onnxscript import opset17 as op

lowpass_filter_width: float = 6.0
rolloff: float = 0.99


@script()
def sinc_resample_kernel(orig_freq: FLOAT, new_freq: FLOAT):
    base_freq = op.Min(orig_freq, new_freq) * rolloff
    width = op.Ceil(lowpass_filter_width * orig_freq / base_freq)

    idx = op.Range(-width, width + orig_freq, 1) / orig_freq
    t = op.Unsqueeze(op.Range(0, -new_freq, -1) / new_freq, -1) + idx
    t = op.Clip(t * base_freq, -lowpass_filter_width, lowpass_filter_width)
    t = t * op.Constant(value=math.pi)

    window = op.Cos(t / (lowpass_filter_width * 2.0)) ** 2
    kernels = op.Where(t == 0.0, 1.0, op.Sin(t) / (t + 1e-20))
    kernels = kernels * window * base_freq / orig_freq

    return op.Unsqueeze(kernels, [1, 2])


@script()
def resample(
    waveforms: FLOAT["batch_size", "N"],
    waveforms_lens: INT64["batch_size"],
    orig_freq: int,
    new_freq: int,
    pads: Sequence[int],
    strides: Sequence[int],
):
    kernel = sinc_resample_kernel(op.Cast(orig_freq, to=FLOAT.dtype), op.Cast(new_freq, to=FLOAT.dtype))
    conv = op.Conv(op.Unsqueeze(waveforms, axes=[1, 2]), kernel, pads=pads, strides=strides)

    resampled = op.Flatten(op.Transpose(conv, perm=(0, 3, 2, 1)))
    resampled_lens = (new_freq * waveforms_lens + orig_freq - 1) / orig_freq

    new_len = (new_freq * op.Shape(waveforms, start=1, end=2)[0] + orig_freq - 1) / orig_freq
    mask = op.Unsqueeze(op.Range(0, new_len, 1), 0) < op.Unsqueeze(resampled_lens, 1)

    return op.Where(mask, resampled[:, :new_len], 0.0), resampled_lens


def kernel_args(orig_freq, new_freq):
    gcd = math.gcd(orig_freq, new_freq)
    orig_freq //= gcd
    new_freq //= gcd
    base_freq = min(orig_freq, new_freq) * rolloff
    width = math.ceil(lowpass_filter_width * orig_freq / base_freq)
    return orig_freq, new_freq, (0, width, 0, width + orig_freq), (1, orig_freq)


@script(doc_string="Resampling waveform to 8 kHz")
def ResamplePreprocessor8(
    waveforms: FLOAT["batch_size", "N"], waveforms_lens: INT64["batch_size"], sample_rate: INT64
) -> tuple[FLOAT["batch_size", "M"], INT64["batch_size"]]:
    if sample_rate == 11_025:
        res, lens = resample(
            waveforms,
            waveforms_lens,
            kernel_args(11_025, 8_000)[0],
            kernel_args(11_025, 8_000)[1],
            kernel_args(11_025, 8_000)[2],
            kernel_args(11_025, 8_000)[3],
        )
    elif sample_rate == 16_000:
        res, lens = resample(
            waveforms,
            waveforms_lens,
            kernel_args(16_000, 8_000)[0],
            kernel_args(16_000, 8_000)[1],
            kernel_args(16_000, 8_000)[2],
            kernel_args(16_000, 8_000)[3],
        )
    elif sample_rate == 22_050:
        res, lens = resample(
            waveforms,
            waveforms_lens,
            kernel_args(22_050, 8_000)[0],
            kernel_args(22_050, 8_000)[1],
            kernel_args(22_050, 8_000)[2],
            kernel_args(22_050, 8_000)[3],
        )
    elif sample_rate == 24_000:
        res, lens = resample(
            waveforms,
            waveforms_lens,
            kernel_args(24_000, 8_000)[0],
            kernel_args(24_000, 8_000)[1],
            kernel_args(24_000, 8_000)[2],
            kernel_args(24_000, 8_000)[3],
        )
    elif sample_rate == 32_000:
        res, lens = resample(
            waveforms,
            waveforms_lens,
            kernel_args(32_000, 8_000)[0],
            kernel_args(32_000, 8_000)[1],
            kernel_args(32_000, 8_000)[2],
            kernel_args(32_000, 8_000)[3],
        )
    elif sample_rate == 44_100:
        res, lens = resample(
            waveforms,
            waveforms_lens,
            kernel_args(44_100, 8_000)[0],
            kernel_args(44_100, 8_000)[1],
            kernel_args(44_100, 8_000)[2],
            kernel_args(44_100, 8_000)[3],
        )
    elif sample_rate == 48_000:
        res, lens = resample(
            waveforms,
            waveforms_lens,
            kernel_args(48_000, 8_000)[0],
            kernel_args(48_000, 8_000)[1],
            kernel_args(48_000, 8_000)[2],
            kernel_args(48_000, 8_000)[3],
        )
    else:
        res, lens = waveforms, waveforms_lens

    resampled, resampled_lens = op.Identity(res), op.Identity(lens)
    return resampled, resampled_lens


@script(doc_string="Resampling waveform to 16 kHz")
def ResamplePreprocessor16(
    waveforms: FLOAT["batch_size", "N"], waveforms_lens: INT64["batch_size"], sample_rate: INT64
) -> tuple[FLOAT["batch_size", "M"], INT64["batch_size"]]:
    if sample_rate == 8_000:
        res, lens = resample(
            waveforms,
            waveforms_lens,
            kernel_args(8_000, 16_000)[0],
            kernel_args(8_000, 16_000)[1],
            kernel_args(8_000, 16_000)[2],
            kernel_args(8_000, 16_000)[3],
        )
    elif sample_rate == 11_025:
        res, lens = resample(
            waveforms,
            waveforms_lens,
            kernel_args(11_025, 16_000)[0],
            kernel_args(11_025, 16_000)[1],
            kernel_args(11_025, 16_000)[2],
            kernel_args(11_025, 16_000)[3],
        )
    elif sample_rate == 22_050:
        res, lens = resample(
            waveforms,
            waveforms_lens,
            kernel_args(22_050, 16_000)[0],
            kernel_args(22_050, 16_000)[1],
            kernel_args(22_050, 16_000)[2],
            kernel_args(22_050, 16_000)[3],
        )
    elif sample_rate == 24_000:
        res, lens = resample(
            waveforms,
            waveforms_lens,
            kernel_args(24_000, 16_000)[0],
            kernel_args(24_000, 16_000)[1],
            kernel_args(24_000, 16_000)[2],
            kernel_args(24_000, 16_000)[3],
        )
    elif sample_rate == 32_000:
        res, lens = resample(
            waveforms,
            waveforms_lens,
            kernel_args(32_000, 16_000)[0],
            kernel_args(32_000, 16_000)[1],
            kernel_args(32_000, 16_000)[2],
            kernel_args(32_000, 16_000)[3],
        )
    elif sample_rate == 44_100:
        res, lens = resample(
            waveforms,
            waveforms_lens,
            kernel_args(44_100, 16_000)[0],
            kernel_args(44_100, 16_000)[1],
            kernel_args(44_100, 16_000)[2],
            kernel_args(44_100, 16_000)[3],
        )
    elif sample_rate == 48_000:
        res, lens = resample(
            waveforms,
            waveforms_lens,
            kernel_args(48_000, 16_000)[0],
            kernel_args(48_000, 16_000)[1],
            kernel_args(48_000, 16_000)[2],
            kernel_args(48_000, 16_000)[3],
        )
    else:
        res, lens = waveforms, waveforms_lens

    resampled, resampled_lens = op.Identity(res), op.Identity(lens)
    return resampled, resampled_lens
