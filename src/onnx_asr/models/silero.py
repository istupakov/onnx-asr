"""Silero VAD implementation."""

import typing
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

from onnx_asr.vad import Vad


class SileroVad(Vad):
    """Silero VAD implementation."""

    CONTEXT_SIZE = 64
    HOP_SIZE = 512

    def __init__(self, model_files: dict[str, Path], **kwargs: typing.Any):
        """Create Silero VAD.

        Args:
            model_files: Dict with paths to model files.
            kwargs: Additional parameters for onnxruntime.InferenceSession.

        """
        self._model = rt.InferenceSession(model_files["model"], **kwargs)

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        return {"model": f"**/model{suffix}.onnx"}

    def _encode(self, waveforms: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        frames = np.lib.stride_tricks.sliding_window_view(waveforms, self.CONTEXT_SIZE + self.HOP_SIZE, axis=-1)[
            :, self.HOP_SIZE - self.CONTEXT_SIZE :: self.HOP_SIZE
        ]

        state = np.zeros((2, frames.shape[0], 128), dtype=np.float32)

        def process(frame: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
            nonlocal state
            output, state = self._model.run(["output", "stateN"], {"input": frame, "state": state, "sr": [self.SAMPLE_RATE]})
            return typing.cast(npt.NDArray[np.float32], output)

        probs = np.empty((waveforms.shape[0], (waveforms.shape[1] + self.HOP_SIZE - 1) // self.HOP_SIZE), dtype=np.float32)
        probs[:, 0] = process(np.pad(waveforms[:, : self.HOP_SIZE], ((0, 0), (self.CONTEXT_SIZE, 0))))
        for i in range(frames.shape[1]):
            probs[:, i + 1] = process(frames[:, i])

        if last_frame := waveforms.shape[1] % self.HOP_SIZE:
            probs[:, -1] = process(
                np.pad(waveforms[:, -last_frame - self.CONTEXT_SIZE :], ((0, 0), (0, self.HOP_SIZE - last_frame)))
            )

        return probs

    def _find_segments(self, waveforms: npt.NDArray[np.float32], **kwargs: float) -> Iterable[Iterable[tuple[int, int]]]:
        def segment(
            probs: npt.NDArray[np.float32], pos_threshold: float = 0.5, neg_threshold: float = 0.35, **kwargs: float
        ) -> Iterable[tuple[int, int]]:
            state = 0
            start = 0
            for i, p in enumerate(probs):
                t = i * self.HOP_SIZE
                if state == 0 and p >= pos_threshold:
                    state = 1
                    start = t
                elif state == 1 and p < neg_threshold:
                    state = 0
                    yield start, t

            if state == 1:
                yield start, len(probs) * self.HOP_SIZE

        return (segment(probs, **kwargs) for probs in self._encode(waveforms))
