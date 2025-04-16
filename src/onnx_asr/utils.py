import numpy as np
import numpy.typing as npt


def pad_list(arrays: list[npt.NDArray[np.float32]], axis: int = 0) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
    lens = np.array([array.shape[axis] for array in arrays])
    max_len = lens.max()

    def pads(array):
        return [(0, max_len - array.shape[axis]) if i == axis else (0, 0) for i in range(array.ndim)]

    return np.stack([np.pad(array, pads(array)) for array in arrays]), lens
