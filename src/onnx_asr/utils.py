import wave
import numpy as np
import numpy.typing as npt


def read_wav(filename: str) -> tuple[npt.NDArray[np.float32], int]:
    """
    Read PCM wav file

    Support PCM_U8, PCM_16, PCM_24 and PCM_32 formats.
    Parameters:
        filename : Path to wav file
    Returns
    -------
    waveform : (frames x channels) numpy array with samples [-1, +1]
    sample_rate : sample rate
    """
    with wave.open(filename, mode="rb") as f:
        data = f.readframes(f.getnframes())
        z = 0
        if f.getsampwidth() == 1:
            buffer = np.frombuffer(data, dtype="u1")
            z = 1
        elif f.getsampwidth() == 3:
            buffer = np.zeros((len(data) // 3, 4), dtype="V1")
            buffer[:, -3:] = np.frombuffer(data, dtype="V1").reshape(-1, f.getsampwidth())
            buffer = buffer.view(dtype="<i4")
        else:
            buffer = np.frombuffer(data, dtype=f"<i{f.getsampwidth()}")

        return np.divide(
            buffer.reshape(f.getnframes(), f.getnchannels()), 2 ** (8 * buffer.itemsize - 1), dtype=np.float32
        ) - z, f.getframerate()  # type: ignore


def pad_list(arrays: list[npt.NDArray[np.float32]], axis: int = 0) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
    lens = np.array([array.shape[axis] for array in arrays])
    max_len = lens.max()

    def pads(array):
        return [(0, max_len - array.shape[axis]) if i == axis else (0, 0) for i in range(array.ndim)]

    return np.stack([np.pad(array, pads(array)) for array in arrays]), lens
