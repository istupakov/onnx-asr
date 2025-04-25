"""Utils for ASR."""

import wave

import numpy as np
import numpy.typing as npt


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


def read_wav_files(waveforms: list[npt.NDArray[np.float32] | str]) -> list[npt.NDArray[np.float32]]:
    """Convert list of waveform or filenames to list of waveforms."""
    results = []
    for x in waveforms:
        if isinstance(x, str):
            waveform, sample_rate = read_wav(x)
            assert sample_rate == 16000, "Supported only 16 kHz sample rate."
            assert waveform.shape[1] == 1, "Supported only mono audio."
            results.append(waveform[:, 0])
        else:
            assert x.ndim == 1, "Waveform must be 1d numpy array."
            results.append(x)

    return results


def pad_list(arrays: list[npt.NDArray[np.float32]], axis: int = 0) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
    """Pad list of Numpy arrays to common length."""
    lens = np.array([array.shape[axis] for array in arrays])
    max_len = lens.max()

    def pads(array: npt.NDArray[np.float32]) -> list[tuple[int, int]]:
        return [(0, max_len - array.shape[axis]) if i == axis else (0, 0) for i in range(array.ndim)]

    return np.stack([np.pad(array, pads(array)) for array in arrays]), lens
