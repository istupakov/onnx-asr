import math

from onnxscript import FLOAT, INT64, script
from onnxscript import opset17 as op

lowpass_filter_width: float = 6.0
rolloff: float = 0.99


@script()
def sinc_resample_kernel(orig_freq: FLOAT, new_freq: FLOAT):
    base_freq = op.Min(orig_freq, new_freq) * rolloff
    width = op.Ceil(lowpass_filter_width * orig_freq / base_freq)

    idx = op.Range(-width, width + orig_freq, 1) / orig_freq
    t = op.Unsqueeze(op.Range(0, -new_freq, -1) / new_freq, axes=[-1]) + idx
    t = op.Clip(t * base_freq, -lowpass_filter_width, lowpass_filter_width)
    t = t * op.Constant(value=math.pi)

    window = op.Cos(t / (lowpass_filter_width * 2.0)) ** 2
    kernels = op.Where(t == 0.0, 1.0, op.Sin(t) / (t + 1e-20))
    kernels = kernels * window * base_freq / orig_freq

    return op.Unsqueeze(kernels, axes=[1, 2])


def create_resampler(orig_freq, new_freq):
    gcd = math.gcd(orig_freq, new_freq)
    orig_freq //= gcd
    new_freq //= gcd
    base_freq = min(orig_freq, new_freq) * rolloff
    width = math.ceil(lowpass_filter_width * orig_freq / base_freq)
    pads = (0, width, 0, width + orig_freq)
    strides = (1, orig_freq)

    @script(doc_string=f"Resampling waveform from {orig_freq * gcd} Hz to {new_freq * gcd} Hz")
    def ResamplePreprocessor(
        waveforms: FLOAT["batch_size", "N"], waveforms_lens: INT64["batch_size"]
    ) -> tuple[FLOAT["batch_size", "M"], INT64["batch_size"]]:
        kernel = sinc_resample_kernel(op.Cast(orig_freq, to=FLOAT.dtype), op.Cast(new_freq, to=FLOAT.dtype))
        conv = op.Conv(op.Unsqueeze(waveforms, axes=[1, 2]), kernel, pads=pads, strides=strides)

        resampled_tmp = op.Flatten(op.Transpose(conv, perm=[0, 3, 2, 1]))
        resampled_lens = (new_freq * waveforms_lens + orig_freq - 1) / orig_freq

        new_len = (new_freq * op.Squeeze(op.Shape(waveforms, start=1, end=2)) + orig_freq - 1) / orig_freq
        mask = op.Unsqueeze(op.Range(0, new_len, 1), axes=[0]) < op.Unsqueeze(resampled_lens, axes=[1])
        resampled = op.Where(
            mask, op.Slice(resampled_tmp, starts=[0], ends=op.Unsqueeze(new_len, axes=[0]), steps=[1], axes=[1]), 0.0
        )
        return resampled, resampled_lens

    return ResamplePreprocessor
