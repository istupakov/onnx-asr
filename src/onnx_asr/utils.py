"""Utils for ASR."""

import wave
from collections.abc import Sequence
from typing import Any, Literal, TypedDict, TypeGuard, get_args

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

SampleRates = Literal[8_000, 16_000, 22_050, 44_100, 48_000]


def is_supported_sample_rate(sample_rate: int) -> TypeGuard[SampleRates]:
    """Sample rate is supported."""
    return sample_rate in get_args(SampleRates)


def is_float32_array(x: object) -> TypeGuard[npt.NDArray[np.float32]]:
    """Numpy array is float32."""
    return isinstance(x, np.ndarray) and x.dtype == np.float32


def is_int32_array(x: object) -> TypeGuard[npt.NDArray[np.int32]]:
    """Numpy array is int32."""
    return isinstance(x, np.ndarray) and x.dtype == np.int32


def is_int64_array(x: object) -> TypeGuard[npt.NDArray[np.int64]]:
    """Numpy array is int64."""
    return isinstance(x, np.ndarray) and x.dtype == np.int64


class SupportedOnlyMonoAudioError(ValueError):
    """Supported only mono audio error."""

    def __init__(self) -> None:
        """Create error."""
        super().__init__("Supported only mono audio.")


class WrongSampleRateError(ValueError):
    """Wrong sample rate error."""

    def __init__(self) -> None:
        """Create error."""
        super().__init__("Supported only 8, 16, 22.05, 44.1 and 48 kHz sample rate.")


class DifferentSampleRatesError(ValueError):
    """Different sample rates error."""

    def __init__(self) -> None:
        """Create error."""
        super().__init__("All sample rates in a batch must be the same.")


class OnnxSessionOptions(TypedDict, total=False):
    """Options for onnxruntime InferenceSession."""

    sess_options: rt.SessionOptions | None
    providers: Sequence[str | tuple[str, dict[Any, Any]]] | None
    provider_options: Sequence[dict[Any, Any]] | None
    cpu_preprocessing: bool


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
        return buffer.reshape(f.getnframes(), f.getnchannels()).astype(np.float32) / max_value - zero_value, f.getframerate()


def read_wav_files(
    waveforms: list[npt.NDArray[np.float32] | str], numpy_sample_rate: SampleRates
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64], SampleRates]:
    """Convert list of waveform or filenames to Numpy array with common length."""
    results = []
    sample_rates = []
    for x in waveforms:
        if isinstance(x, str):
            waveform, sample_rate = read_wav(x)
            if waveform.shape[1] != 1:
                raise SupportedOnlyMonoAudioError()
            results.append(waveform[:, 0])
            sample_rates.append(sample_rate)
        else:
            if x.ndim != 1:
                raise SupportedOnlyMonoAudioError()
            results.append(x)
            sample_rates.append(numpy_sample_rate)

    if len(set(sample_rates)) > 1:
        raise DifferentSampleRatesError()

    if is_supported_sample_rate(sample_rates[0]):
        return *pad_list(results), sample_rates[0]
    raise WrongSampleRateError()


def pad_list(arrays: list[npt.NDArray[np.float32]]) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
    """Pad list of Numpy arrays to common length."""
    lens = np.array([array.shape[0] for array in arrays], dtype=np.int64)

    result = np.zeros((len(arrays), lens.max()), dtype=np.float32)
    for i, x in enumerate(arrays):
        result[i, : x.shape[0]] = x[: min(x.shape[0], result.shape[1])]

    return result, lens
