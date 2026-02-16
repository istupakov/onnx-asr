"""ASR preprocessor implementations in NumPy."""

from __future__ import annotations

from importlib.resources import as_file, files

import numpy as np
import numpy.typing as npt

import onnx_asr.preprocessors


class _NumpyPreprocessor:
    def __init__(self, name: str):
        """Create preprocessor.

        Args:
            name: Preprocessor name.

        """
        with (
            as_file(files(onnx_asr.preprocessors).joinpath("data").joinpath("fbanks.npz")) as file,
            np.load(file) as data,
        ):
            self._melscale_fbanks = data[name]
            if name == "gigaam_v3":
                self._window = data["gigaam_v3_window"]


class GigaamPreprocessorNumpy(_NumpyPreprocessor):
    """GigaAM preprocessor implementation in NumPy."""

    _sample_rate = 16_000
    _hop_length = _sample_rate // 100
    _clamp_min = 1e-9
    _clamp_max = 1e9

    def __init__(self, name: str):  # noqa: D107
        assert name in ("gigaam_v2", "gigaam_v3")
        super().__init__(name)
        self._v2 = name == "gigaam_v2"
        self._n_fft = self._sample_rate // (40 if self._v2 else 50)
        self._win_length = self._n_fft
        if self._v2:
            self._window = np.hanning(self._win_length + 1)[:-1]

    def __call__(
        self, waveforms: npt.NDArray[np.float32], waveforms_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        """Convert waveforms to model features."""
        if self._v2:
            waveforms = np.pad(waveforms, ((0, 0), (self._n_fft // 2, self._n_fft // 2)), mode="reflect")

        strided_input = np.lib.stride_tricks.sliding_window_view(waveforms, self._win_length, axis=1)[
            :, :: self._hop_length
        ]
        strided_input = strided_input * self._window
        spectrum = np.abs(np.fft.rfft(strided_input, self._n_fft)) ** 2

        mel_energies = np.matmul(spectrum, self._melscale_fbanks).astype(np.float32)

        return np.log(np.clip(mel_energies, self._clamp_min, self._clamp_max)).transpose(0, 2, 1), (
            waveforms_lens - (0 if self._v2 else self._win_length)
        ) // self._hop_length + 1


class KaldiPreprocessorNumpy(_NumpyPreprocessor):
    """Kaldi preprocessor implementation with NumPy."""

    _n_fft = 512
    _win_length = 400
    _hop_length = 160
    _dither = 0.0
    _remove_dc_offset = True
    _preemphasis_coefficient = 0.97
    _float_eps = float(np.finfo(np.float32).eps)

    def __init__(self, name: str):  # noqa: D107
        assert name in ("kaldi", "wespeaker")
        super().__init__(name)
        if name == "kaldi":
            self._snip_edges = False
            self._window = np.hanning(self._win_length).astype(np.float32) ** 0.85
        else:
            self._snip_edges = True
            self._window = np.hamming(self._win_length).astype(np.float32)

    def _symmetric_pad(
        self, waveforms: npt.NDArray[np.float32], waveforms_lens: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float32]:
        pad_left = self._win_length // 2 - self._hop_length // 2
        pad_right = self._win_length // 2
        res = np.pad(waveforms, ((0, 0), (pad_left, pad_right)), mode="symmetric")
        if waveforms.shape[0] == 1:
            return res

        for i in range(waveforms.shape[0]):
            tail = res[i, pad_left + waveforms_lens[i] :]
            tail[:pad_right] = waveforms[i, waveforms_lens[i] - pad_right : waveforms_lens[i]][::-1]
            tail[pad_right:] = 0
        return res

    def __call__(
        self, waveforms: npt.NDArray[np.float32], waveforms_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        """Convert waveforms to model features."""
        if not self._snip_edges:
            waveforms = self._symmetric_pad(waveforms, waveforms_lens)
            features_lens = (waveforms_lens + self._hop_length // 2) // self._hop_length
        else:
            features_lens = 1 + (waveforms_lens - self._win_length) // self._hop_length

        strided_input = np.lib.stride_tricks.sliding_window_view(waveforms, self._win_length, axis=1)[
            :, :: self._hop_length
        ]

        if self._dither != 0.0:
            rng = np.random.default_rng()
            strided_input = strided_input + self._dither * rng.standard_normal(strided_input.shape).astype(np.float32)

        if self._remove_dc_offset:
            strided_input = strided_input - np.mean(strided_input, axis=-1, keepdims=True)

        if self._preemphasis_coefficient != 0.0:
            offset_strided_input = np.pad(strided_input, ((0, 0), (0, 0), (1, 0)), mode="edge")
            strided_input = strided_input - self._preemphasis_coefficient * offset_strided_input[..., :-1]

        strided_input = strided_input * self._window
        spectrum = np.abs(np.fft.rfft(strided_input, self._n_fft)) ** 2
        mel_energies = np.matmul(spectrum, self._melscale_fbanks).astype(np.float32)

        features = np.log(np.maximum(mel_energies, np.finfo(np.float32).eps))
        if features.shape[0] > 0:
            features[np.arange(features.shape[1]) >= features_lens[:, None]] = 0

        return features, features_lens


class NemoPreprocessorNumpy(_NumpyPreprocessor):
    """Nemo preprocessor implementation with NumPy."""

    _n_fft = 512
    _win_length = 400
    _hop_length = 160
    _preemph = 0.97
    _log_zero_guard_value = float(2**-24)

    def __init__(self, name: str):  # noqa: D107
        assert name.startswith("nemo")
        super().__init__(name)

    def __call__(
        self, waveforms: npt.NDArray[np.float32], waveforms_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        """Convert waveforms to model features."""
        if self._preemph != 0.0:
            waveforms = waveforms - self._preemph * np.pad(waveforms, ((0, 0), (1, 0)))[:, :-1]
            waveforms[np.arange(waveforms.shape[-1]) >= waveforms_lens[:, None]] = 0

        waveforms = np.pad(waveforms, ((0, 0), (self._n_fft // 2, self._n_fft // 2)))
        strided_input = np.lib.stride_tricks.sliding_window_view(waveforms, self._n_fft, axis=1)[:, :: self._hop_length]
        strided_input = strided_input * np.pad(
            np.hanning(self._win_length), ((self._n_fft - self._win_length) // 2, (self._n_fft - self._win_length) // 2)
        )
        spectrogram = np.abs(np.fft.rfft(strided_input, self._n_fft)) ** 2
        mel_spectrogram = np.matmul(spectrogram, self._melscale_fbanks)
        log_mel_spectrogram = np.log(mel_spectrogram + self._log_zero_guard_value)

        features_lens = waveforms_lens // self._hop_length
        mask = np.arange(log_mel_spectrogram.shape[1])[None, :, None] < features_lens[:, None, None]
        zero = np.float32(0)
        mean = np.where(mask, log_mel_spectrogram, zero).sum(axis=1, keepdims=True) / features_lens[:, None, None]
        var = np.where(mask, (log_mel_spectrogram - mean) ** 2, zero).sum(axis=1, keepdims=True) / (
            features_lens[:, None, None] - 1
        )
        features = np.where(mask, (log_mel_spectrogram - mean) / (np.sqrt(var) + 1e-5), zero)
        return features.transpose(0, 2, 1).astype(np.float32), features_lens


class WhisperPreprocessorNumpy(_NumpyPreprocessor):
    """Whisper preprocessor implementation with NumPy."""

    _sample_rate = 16_000
    _chunk_length = 30
    _n_fft = 400
    _win_length = 400
    _hop_length = 160
    _clamp_min = 1e-10

    def __init__(self, name: str):  # noqa: D107
        assert name.startswith("whisper")
        super().__init__(name)

    def __call__(
        self, waveforms: npt.NDArray[np.float32], waveforms_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        """Convert waveforms to model features."""
        waveforms = waveforms[:, : self._chunk_length * self._sample_rate]
        waveforms = np.pad(waveforms, ((0, 0), (0, self._chunk_length * self._sample_rate - waveforms.shape[-1])))
        waveforms = np.pad(waveforms, ((0, 0), (self._n_fft // 2, self._n_fft // 2)), mode="reflect")

        strided_input = np.lib.stride_tricks.sliding_window_view(waveforms, self._win_length, axis=1)[
            :, :: self._hop_length
        ]
        strided_input = strided_input * np.hanning(self._win_length + 1)[:-1]
        spectrum = np.abs(np.fft.rfft(strided_input, self._n_fft)[:, :-1]) ** 2

        mel_spectrogram = np.matmul(spectrum, self._melscale_fbanks).astype(np.float32)
        log_mel_spectrogram = np.log10(np.maximum(mel_spectrogram, self._clamp_min))
        features = (np.maximum(log_mel_spectrogram, log_mel_spectrogram.max() - 8.0) + 4.0) / 4.0
        return features.transpose(0, 2, 1), np.full_like(
            waveforms_lens, self._chunk_length * self._sample_rate // self._hop_length
        )
