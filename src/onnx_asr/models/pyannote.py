
from onnx_asr.vad import Vad
from onnx_asr.onnx import OnnxSessionOptions, TensorRtOptions
from onnx_asr.utils import is_float32_array

import onnxruntime as rt

from pathlib import Path

from collections.abc import Iterable, Iterator
from typing import Literal

import numpy as np
import numpy.typing as npt

from itertools import permutations, chain


class PyAnnoteVad(Vad):
    """ Pyannote VAD implementation. """

    INF = 10**15
    RECEPTIVE_FIELD = 991
    STRIDE = 270

    def __init__(self, model_files: dict[str, Path], onnx_options: OnnxSessionOptions):
        """ Create Pyannote VAD """

        self._model = rt.InferenceSession(model_files["model"], **onnx_options)
        self._num_windows = 0
        
    @staticmethod
    def _get_excluded_providers() -> list[str]:
        return TensorRtOptions.get_provider_names()

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        return {"model": f"**/model{suffix}.onnx"}

    @staticmethod
    def _sample2frame(L_in: int) -> int:
        # receptive field: 991
        # stride: 270
        return (L_in - PyAnnoteVad.RECEPTIVE_FIELD) // PyAnnoteVad.STRIDE + 1

    @staticmethod
    def _frame2sample(L_in: int) -> int:
        # receptive field: 991
        # stride: 270
        return (L_in - 1) * PyAnnoteVad.STRIDE + PyAnnoteVad.RECEPTIVE_FIELD

    def _encode(
        self,
        waveforms: npt.NDArray[np.float32],
        window_size: int,
        overlap: int,
        **kwargs
    ) -> Iterator[npt.NDArray[np.float32]]:
        """
        Args:
            waveforms: audio samples with shape (batch_size, num_samples)
            window_size: number of window samples
            overlap: number of overlap samples
        """

        # 10s sliding window and 5s overlap
        windows = np.lib.stride_tricks.sliding_window_view(waveforms, window_size, axis=-1)[
            :, :: (window_size - overlap), :
        ] # This will drop the last window if its length is less than the overlap (5s), be careful

        self._num_windows = windows.shape[1]
        if last_window_size := waveforms.shape[1] % (window_size - overlap):
            self._num_windows += 1

        def process(window: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
            """
            Input:
                window: (batch_size, window_size) 
            Output:
                batch of shape(batch_size, num_frames(589 for 10s), 7)
                { no-speech }, { spk1 }, { spk2 }, { spk3 }, { spk1 + spk2 }, { spk1 + spk3 }, { spk2 + spk3 }
            """
            output = np.exp(self._model.run(['logits'], {'input_values': window[:, None, :]})[0])
            assert is_float32_array(output)
            return output
        
        for i in range(windows.shape[1]):
            yield process(windows[:, i, :])

        if last_window_size := waveforms.shape[1] % (window_size - overlap):
            yield process(np.pad(waveforms[:, -(last_window_size + overlap) : ],
                                 ((0, 0), (0, (window_size - overlap - last_window_size)))
                            ))
            
    def _decode(
        self,
        windows: Iterator[npt.NDArray[np.float32]],
        window_size:int,
        overlap: int,
        **kwargs
    ) -> Iterator[tuple[int, npt.NDArray[np.float32]]]:
        """
        Args:
            windows: Iterator of window with shape (num_frames, num_spk) where num_spk = 7
            window_size: number of window samples
            overlap: number of overlap samples
        """
        def reorder(window: npt.NDArray[np.float32],
                    chunk: npt.NDArray[np.float32]
        ) -> npt.NDArray[np.float32]:
            """ Make sure all the windows have the correct speaker order """
            perms = [np.array(perm).T for perm in permutations(window.T)]
            diffs = np.sum(np.abs(np.sum(np.array(perms)[ : , : overlap_len] - chunk, axis=1)), axis=1)
            return perms[np.argmin(diffs)]
        
        def fuse(window, chunk):
            """ Combine overlapping chunks and take an average of their probs """
            window[ : ,  : ] = (window[ : ,  : ] + chunk[ : ,  : ]) / 2

        def merge_spk_probs(window):
            # for each spk add all probs up
            window[ :, 1] += window[ :, 4] + window[ :, 5]
            window[ :, 2] += window[ :, 4] + window[ :, 6]
            window[ :, 3] += window[ :, 5] + window[ :, 6]

            # dump other probs
            return window[ :, :4]

        overlap_len = self._sample2frame(overlap)
        overlap_chunk = np.zeros((overlap_len, 4))
        for i, window in enumerate(windows, 1):
            window = merge_spk_probs(window)
            if i == 1:
                overlap_chunk = window[-overlap_len : ]
                yield 0, window[ : -overlap_len]
            elif i > 1 and i < self._num_windows:
                #window[ : , 1:] = reorder(window[ : , 1: ], overlap_chunk[ : , 1: ])
                fuse(window[ : overlap_len], overlap_chunk)
                overlap_chunk = window[-overlap_len : ]
                yield (i - 1) * (window_size - overlap), window[ : -overlap_len]
            else:
                #window[ : , 1:] = reorder(window[ : , 1:], overlap_chunk[ : , 1: ])
                fuse(window[ : overlap_len], overlap_chunk)
                yield (i - 1) * (window_size - overlap), window

    def _find_segments(
        self,
        decoding: Iterable[tuple[int, npt.NDArray[np.float32]]],
        *,
        threshold: float = 0.5,
        neg_threshold: float | None = None,
        **kwargs: float
    ) -> Iterator[tuple[int, int]]:
        """
        Args:
            decoding: a Iterable of (begin and window)
        """
        if neg_threshold is None:
            neg_threshold = threshold - 0.15

        offset = 135 # half of the stride

        state = 0
        start = 0
        last_begin = 0
        last_i = 0
        for begin, window in decoding:
            last_begin = begin
            for frame_idx, p in enumerate(window):
                last_i = frame_idx
                if state == 0 and p > threshold:
                    state = 1
                    start = begin + self._frame2sample(frame_idx) + offset
                if state == 1 and p < neg_threshold:
                    state = 0
                    yield start, begin + self._frame2sample(frame_idx) + offset

        if state == 1:
            yield start, last_begin + self._frame2sample(last_i) + offset

        
    def _merge_segments(
        self,
        segments: Iterator[tuple[int, int]],
        waveform_len: int,
        sample_rate: int,
        *,
        min_speech_duration_ms: float = 200,
        max_speech_duration_s: float = 20,
        min_silence_duration_ms: float = 100,
        speech_pad_ms: float = 50,
        **kwargs: float
    ) -> Iterator[tuple[int, int]]:
        speech_pad = int(speech_pad_ms * sample_rate // 1000)
        min_speech_duration = int(min_speech_duration_ms * sample_rate // 1000) - 2 * speech_pad
        max_speech_duration = int(max_speech_duration_s * sample_rate) - 2 * speech_pad
        min_silence_duration = int(min_silence_duration_ms * sample_rate // 1000) + 2 * speech_pad

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

    def segment_batch(
        self,
        waveforms: npt.NDArray[np.float32],
        waveforms_len: npt.NDArray[np.int64],
        sample_rate: Literal[16_000],
        **kwargs: float
    ) -> Iterator[Iterator[tuple[int, int]]]:
        """
        Args:
            waveforms: audio samples with shape (batch_size, num_samples)
            waveforms_len: ints with shape( batch_size, 1 )
            sample_rate: Literal[16_000]
        """
        window_size = 10 * sample_rate
        overlap = 5 * sample_rate

        def segment(decoding: Iterator[npt.NDArray[np.float32]], waveform_len: np.int64, **kwargs: float) -> Iterator[tuple[int, int]]:
            return self._merge_segments(self._find_segments(decoding, **kwargs), waveform_len, sample_rate, **kwargs)

        encoding = self._encode(waveforms, window_size, overlap, **kwargs)
        if len(waveforms) == 1:
            decoding = ((begin, 1-window[: , 0]) # only put the first {no-speech} prob in. prob_speech = 1 - prob(no-speech)
                        for begin, window in self._decode((batch_windows[0] for batch_windows in encoding), window_size, overlap))
            yield segment(decoding, waveforms_len[0], **kwargs)
        else:
            yield from (
                segment(((begin, 1-window[:, 0]) for begin, window in self._decode(undecode_windows, window_size, overlap)), waveform_len, **kwargs)
                for undecode_windows, waveform_len in zip(zip(*encoding, strict=True), waveforms_len)
            ) 

