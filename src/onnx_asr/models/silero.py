"""Silero VAD implementation."""

import typing
from collections.abc import Iterable, Iterator
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

    def _encode(self, waveforms: npt.NDArray[np.float32]) -> Iterator[npt.NDArray[np.float32]]:
        frames = np.lib.stride_tricks.sliding_window_view(waveforms, self.CONTEXT_SIZE + self.HOP_SIZE, axis=-1)[
            :, self.HOP_SIZE - self.CONTEXT_SIZE :: self.HOP_SIZE
        ]

        state = np.zeros((2, frames.shape[0], 128), dtype=np.float32)

        def process(frame: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
            nonlocal state
            output, state = self._model.run(["output", "stateN"], {"input": frame, "state": state, "sr": [self.SAMPLE_RATE]})
            return typing.cast(npt.NDArray[np.float32], output)

        yield process(np.pad(waveforms[:, : self.HOP_SIZE], ((0, 0), (self.CONTEXT_SIZE, 0))))
        for i in range(frames.shape[1]):
            yield process(frames[:, i])

        if last_frame := waveforms.shape[1] % self.HOP_SIZE:
            yield process(np.pad(waveforms[:, -last_frame - self.CONTEXT_SIZE :], ((0, 0), (0, self.HOP_SIZE - last_frame))))

    def _find_segments(self, waveforms: npt.NDArray[np.float32], **kwargs: float) -> Iterator[Iterator[tuple[int, int]]]:
        def segment(
            probs: Iterable[np.float32], pos_threshold: float = 0.5, neg_threshold: float = 0.35, **kwargs: float
        ) -> Iterator[tuple[int, int]]:
            state = 0
            start = 0
            for i, p in enumerate(probs):
                if state == 0 and p >= pos_threshold:
                    state = 1
                    start = i * self.HOP_SIZE
                elif state == 1 and p < neg_threshold:
                    state = 0
                    yield start, i * self.HOP_SIZE

            if state == 1:
                yield start, waveforms.shape[1]

        if len(waveforms) == 1:
            yield segment((probs[0] for probs in self._encode(waveforms)), **kwargs)
        else:
            yield from (segment(probs, **kwargs) for probs in zip(*self._encode(waveforms), strict=True))

    def _process_segments(
        self,
        segments: Iterator[tuple[int, int]],
        max_end: int,
        min_speech_duration: float = 250,
        min_silence_duration: float = 100,
        speech_pad: float = 30,
        **kwargs: float,
    ) -> Iterator[tuple[int, int]]:
        min_speech_duration *= self.SAMPLE_RATE // 1000
        min_silence_duration *= self.SAMPLE_RATE // 1000
        speech_pad = int(self.SAMPLE_RATE * speech_pad // 1000)

        cur_start, cur_end = -self.SAMPLE_RATE, -self.SAMPLE_RATE
        for start, end in segments:
            if start - cur_end < min_silence_duration + 2 * speech_pad:
                cur_end = end
            else:
                if cur_end - cur_start > min_speech_duration - 2 * speech_pad:
                    yield max(cur_start - speech_pad, 0), min(cur_end + speech_pad, max_end)
                cur_start, cur_end = start, end

        if cur_end - cur_start > min_speech_duration:
            yield max(cur_start - speech_pad, 0), min(cur_end + speech_pad, max_end)

    def segment_batch(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64], **kwargs: float
    ) -> Iterator[Iterator[tuple[int, int]]]:
        """Segment waveforms batch."""
        return (
            self._process_segments(segments, max_end, **kwargs)
            for segments, max_end in zip(self._find_segments(waveforms, **kwargs), waveforms_len, strict=True)
        )
