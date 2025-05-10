"""Silero VAD implementation."""

from collections.abc import Iterable, Iterator
from itertools import chain
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

from onnx_asr.utils import OnnxSessionOptions, is_float32_array
from onnx_asr.vad import Vad


class SileroVad(Vad):
    """Silero VAD implementation."""

    CONTEXT_SIZE = 64
    HOP_SIZE = 512
    INF = 10**15

    def __init__(self, model_files: dict[str, Path], onnx_options: OnnxSessionOptions):
        """Create Silero VAD.

        Args:
            model_files: Dict with paths to model files.
            onnx_options: Options for onnxruntime InferenceSession.

        """
        self._model = rt.InferenceSession(model_files["model"], **onnx_options)

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
            output, new_state = self._model.run(["output", "stateN"], {"input": frame, "state": state, "sr": [self.SAMPLE_RATE]})
            assert is_float32_array(output) and is_float32_array(new_state)
            state = new_state
            return output[:, 0]

        yield process(np.pad(waveforms[:, : self.HOP_SIZE], ((0, 0), (self.CONTEXT_SIZE, 0))))

        for i in range(frames.shape[1]):
            yield process(frames[:, i])

        if last_frame := waveforms.shape[1] % self.HOP_SIZE:
            yield process(np.pad(waveforms[:, -last_frame - self.CONTEXT_SIZE :], ((0, 0), (0, self.HOP_SIZE - last_frame))))

    def _find_segments(
        self, probs: Iterable[np.float32], threshold: float = 0.5, neg_threshold: float | None = None, **kwargs: float
    ) -> Iterator[tuple[int, int]]:
        if neg_threshold is None:
            neg_threshold = threshold - 0.15

        state = 0
        start = 0
        for i, p in enumerate(chain(probs, (np.float32(0),))):
            if state == 0 and p >= threshold:
                state = 1
                start = i * self.HOP_SIZE
            elif state == 1 and p < neg_threshold:
                state = 0
                yield start, i * self.HOP_SIZE

    def _merge_segments(
        self,
        segments: Iterator[tuple[int, int]],
        waveform_len: int,
        min_speech_duration_ms: float = 250,
        max_speech_duration_s: float = 20,
        min_silence_duration_ms: float = 100,
        speech_pad_ms: float = 30,
        **kwargs: float,
    ) -> Iterator[tuple[int, int]]:
        speech_pad = int(speech_pad_ms * self.SAMPLE_RATE // 1000)
        min_speech_duration = int(min_speech_duration_ms * self.SAMPLE_RATE // 1000) - 2 * speech_pad
        max_speech_duration = int(max_speech_duration_s * self.SAMPLE_RATE) - 2 * speech_pad
        min_silence_duration = int(min_silence_duration_ms * self.SAMPLE_RATE // 1000) + 2 * speech_pad

        cur_start, cur_end = -self.INF, -self.INF
        for start, end in chain(segments, ((waveform_len, waveform_len), (self.INF, self.INF))):
            if start - cur_end < min_silence_duration and end - cur_start < max_speech_duration:
                cur_end = end
            else:
                if cur_end - cur_start > min_speech_duration:
                    yield max(cur_start - speech_pad, 0), min(cur_end + speech_pad, waveform_len)
                while end - start > max_speech_duration:
                    yield max(start - speech_pad, 0), start + max_speech_duration - speech_pad
                    start += max_speech_duration
                cur_start, cur_end = start, end

    def _segment(self, probs: Iterable[np.float32], waveform_len: np.int64, **kwargs: float) -> Iterator[tuple[int, int]]:
        return self._merge_segments(self._find_segments(probs, **kwargs), int(waveform_len), **kwargs)

    def segment_batch(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64], **kwargs: float
    ) -> Iterator[Iterator[tuple[int, int]]]:
        """Segment waveforms batch."""
        encoding = self._encode(waveforms)
        if len(waveforms) == 1:
            yield self._segment((probs[0] for probs in encoding), waveforms_len[0], **kwargs)
        else:
            yield from (
                self._segment(probs, waveform_len, **kwargs)
                for probs, waveform_len in zip(zip(*encoding, strict=True), waveforms_len, strict=True)
            )
