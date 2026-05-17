"""STFT computed as a fixed 1d convolution.

op.STFT is not supported by the TensorRT and CUDA execution providers, so any
preprocessor graph that uses it falls back to CPU. The discrete Fourier
transform can instead be written as a 1d convolution with a fixed kernel (the
cos/sin Fourier basis multiplied by the analysis window). The convolution and
the ops around it (Reshape, ReduceSumSquare, Transpose) are supported by every
execution provider, so the resulting graph can run fully on GPU / TensorRT.
"""

import numpy as np
import numpy.typing as npt
from onnxscript import FLOAT, script
from onnxscript import opset17 as op

hop_length = 160


def stft_conv_weights(window: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Build Conv weights that compute a windowed DFT.

    Args:
        window: Analysis window, already zero-padded to the FFT size by the
            caller (its length is used as ``n_fft``).

    Returns:
        Kernel of shape ``[2 * (n_fft // 2 + 1), 1, n_fft]`` stacking the real
        (cos) and imaginary (-sin) parts of the Fourier basis. Used with Conv
        (stride = ``hop_length``) it reproduces ``op.STFT``.

    """
    n_fft = window.shape[0]
    indices = np.arange(n_fft // 2 + 1)[:, np.newaxis] * np.arange(n_fft)[np.newaxis, :]
    angle = 2 * np.pi * indices / n_fft
    basis = np.concatenate([np.cos(angle), -np.sin(angle)]) * window
    return basis[:, np.newaxis, :].astype(np.float32)


@script()
def conv_power_spectrogram(waveforms: FLOAT["batch_size", "N"], conv_weights: FLOAT["channels", 1, "n_fft"]):
    """Power spectrogram [batch_size, frames, n_bins] via a Conv-based STFT.

    Drop-in replacement for ``op.STFT`` followed by ``ReduceSumSquare`` over the
    real/imaginary axis. ``conv_weights`` is built by :func:`stft_conv_weights`.
    """
    image = op.Conv(op.Unsqueeze(waveforms, axes=[1]), conv_weights, strides=[hop_length])
    n_bins = op.Shape(conv_weights, start=0, end=1) / 2
    shape = op.Concat(op.Constant(value=[0, 2]), n_bins, op.Constant(value=[-1]), axis=0)
    spectrogram = op.ReduceSumSquare(op.Reshape(image, shape), axes=[1], keepdims=0)
    return op.Transpose(spectrogram, perm=[0, 2, 1])
