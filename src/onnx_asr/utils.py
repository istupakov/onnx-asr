"""Utils for ASR."""

import wave
from pathlib import Path
from typing import Literal, TypeGuard, cast, get_args

import numpy as np
import numpy.typing as npt

SampleRates = Literal[8_000, 11_025, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000]
"""Supported sample rates."""


def is_supported_sample_rate(sample_rate: int) -> TypeGuard[SampleRates]:
    """Sample rate is supported."""
    return sample_rate in get_args(SampleRates)


def is_float16_array(x: object) -> TypeGuard[npt.NDArray[np.float16]]:
    """Numpy array is float16."""
    return isinstance(x, np.ndarray) and x.dtype == np.float16


def is_float32_array(x: object) -> TypeGuard[npt.NDArray[np.float32]]:
    """Numpy array is float32."""
    return isinstance(x, np.ndarray) and x.dtype == np.float32


def is_int32_array(x: object) -> TypeGuard[npt.NDArray[np.int32]]:
    """Numpy array is int32."""
    return isinstance(x, np.ndarray) and x.dtype == np.int32


def is_int64_array(x: object) -> TypeGuard[npt.NDArray[np.int64]]:
    """Numpy array is int64."""
    return isinstance(x, np.ndarray) and x.dtype == np.int64


class ModelLoadingError(Exception):
    """Model loading error."""


class ModelNotSupportedError(ModelLoadingError, ValueError):
    """Model not supported error."""

    def __init__(self, model: str):
        """Create error."""
        super().__init__(f"Model '{model}' not supported!")


class ModelPathNotDirectoryError(ModelLoadingError, NotADirectoryError):
    """Model path not a directory error."""

    def __init__(self, path: str | Path):
        """Create error."""
        super().__init__(f"The path '{path}' is not a directory.")


class ModelFileNotFoundError(ModelLoadingError, FileNotFoundError):
    """Model file not found error."""

    def __init__(self, filename: str | Path, path: str | Path):
        """Create error."""
        super().__init__(f"File '{filename}' not found in path '{path}'.")


class MoreThanOneModelFileFoundError(ModelLoadingError, OSError):
    """More than one model file found error."""

    def __init__(self, filename: str | Path, path: str | Path):
        """Create error."""
        super().__init__(f"Found more than 1 file '{filename}' in path '{path}'.")


class NoModelNameOrPathSpecifiedError(ModelLoadingError, ValueError):
    """No model name or path specified error."""

    def __init__(self) -> None:
        """Create error."""
        super().__init__("If the path is not specified, you must specify a specific model name.")


class InvalidModelTypeInConfigError(ModelLoadingError, ValueError):
    """Invalid model type in config error."""

    def __init__(self, model_type: str) -> None:
        """Create error."""
        super().__init__(f"Invalid model type '{model_type}' in config.json.")


class AudioLoadingError(ValueError):
    """Audio loading error."""


class SupportedOnlyMonoAudioError(AudioLoadingError):
    """Supported only mono audio error."""

    def __init__(self) -> None:
        """Create error."""
        super().__init__(
            "Supported only mono audio. Use the 'channel' parameter to select a channel or 'mean' to average channels."
        )


class WrongSampleRateError(AudioLoadingError):
    """Wrong sample rate error."""

    def __init__(self) -> None:
        """Create error."""
        super().__init__(f"Supported only {get_args(SampleRates)} sample rates.")


class DifferentSampleRatesError(AudioLoadingError):
    """Different sample rates error."""

    def __init__(self) -> None:
        """Create error."""
        super().__init__("All sample rates in a batch must be the same.")


def read_wav(filename: str) -> tuple[npt.NDArray[np.float32], int]:
    """Read PCM wav file to Numpy array."""
    with wave.open(filename, mode="rb") as f:
        data = f.readframes(f.getnframes())
        zero_value = 0
        if f.getsampwidth() == 1:
            buffer = np.frombuffer(data, dtype="u1")
            zero_value = 1
        elif f.getsampwidth() == 3:
            buffer = np.zeros((len(data) // 3, 4), dtype="V1")
            buffer[:, -3:] = np.frombuffer(data, dtype="V1").reshape(-1, f.getsampwidth())
            buffer = buffer.view(dtype="<i4")
        else:
            buffer = np.frombuffer(data, dtype=f"<i{f.getsampwidth()}")

        max_value = 2 ** (8 * buffer.itemsize - 1)
        return buffer.reshape(f.getnframes(), f.getnchannels()).astype(
            np.float32
        ) / max_value - zero_value, f.getframerate()


def _select_channel(
    waveform: npt.NDArray[np.float32], channel: int | Literal["mean"] | None
) -> npt.NDArray[np.float32]:
    if channel is not None:
        return waveform.mean(axis=-1) if channel == "mean" else waveform[:, channel]
    if waveform.shape[1] == 1:
        return waveform[:, 0]
    raise SupportedOnlyMonoAudioError


def read_wav_files(
    waveforms: list[npt.NDArray[np.float32] | str | Path],
    numpy_sample_rate: SampleRates = 16_000,
    channel: int | Literal["mean"] | None = None,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64], SampleRates]:
    """Convert list of waveform or filenames to Numpy array with common length."""
    results = []
    sample_rates = []
    for x in waveforms:
        if isinstance(x, (str, Path)):
            waveform, sample_rate = read_wav(str(x))
            results.append(_select_channel(waveform, channel))
            sample_rates.append(sample_rate)
        else:
            x = x.squeeze()
            results.append(_select_channel(x, channel) if x.ndim != 1 else x)
            sample_rates.append(numpy_sample_rate)

    if len(set(sample_rates)) > 1:
        raise DifferentSampleRatesError

    if is_supported_sample_rate(sample_rates[0]):
        return *pad_list(results), sample_rates[0]
    raise WrongSampleRateError


def pad_list(arrays: list[npt.NDArray[np.float32]]) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
    """Pad list of Numpy arrays to common length."""
    lens = np.array([array.shape[0] for array in arrays], dtype=np.int64)

    result = np.zeros((len(arrays), lens.max()), dtype=np.float32)
    for i, x in enumerate(arrays):
        result[i, : x.shape[0]] = x[: min(x.shape[0], result.shape[1])]

    return result, lens


def log_softmax(logits: npt.NDArray[np.float32], axis: int | None = None) -> npt.NDArray[np.float32]:
    """Calculate logarithm of softmax."""
    tmp = logits - np.max(logits, axis=axis)
    tmp -= np.log(np.sum(np.exp(tmp), axis=axis))
    return cast(npt.NDArray[np.float32], tmp)
