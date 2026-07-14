from collections.abc import Iterator
from typing import Literal, get_args

import numpy as np
import numpy.typing as npt
import pytest

import onnx_asr
from onnx_asr.adapters import TextResultsAsrAdapter
from onnx_asr.loader import VadNames
from onnx_asr.vad import BaseVad, SegmentResult, TimestampedSegmentResult, Vad


class _TestVad(BaseVad):
    def segment_batch(
        self,
        waveforms: npt.NDArray[np.float32],
        waveforms_len: npt.NDArray[np.int64],
        sample_rate: Literal[8_000, 16_000],
        **kwargs: float,
    ) -> Iterator[Iterator[tuple[int, int]]]:
        return iter(())


@pytest.fixture
def base_vad() -> BaseVad:
    return _TestVad()


def test_merge_segments_ignores_sentinels(base_vad: BaseVad) -> None:
    sample_rate = 16_000
    waveform_len = 30 * sample_rate

    result = list(
        base_vad._merge_segments(
            iter([(3 * sample_rate, 28 * sample_rate)]),
            waveform_len,
            sample_rate,
            min_speech_duration_ms=150,
            max_speech_duration_s=30,
            min_silence_duration_ms=1200,
            speech_pad_ms=110,
        )
    )

    assert result == [(46_240, 449_760)]
    assert all(0 <= start < end <= waveform_len for start, end in result)


def test_merge_segments_returns_nothing_for_silence(base_vad: BaseVad) -> None:
    result = list(
        base_vad._merge_segments(
            iter([]),
            30 * 16_000,
            16_000,
            min_speech_duration_ms=150,
            speech_pad_ms=110,
        )
    )

    assert result == []


def test_merge_segments_preserves_max_duration_splitting(base_vad: BaseVad) -> None:
    result = list(
        base_vad._merge_segments(
            iter([(100, 900)]),
            1_000,
            1_000,
            max_speech_duration_s=0.5,
            speech_pad_ms=50,
        )
    )

    assert result == [(50, 550), (450, 950)]


@pytest.fixture(scope="module")
def model() -> TextResultsAsrAdapter:
    return onnx_asr.load_model("whisper-base", quantization="int8")


@pytest.fixture(scope="module", params=get_args(VadNames))
def vad(request: pytest.FixtureRequest) -> Vad:
    return onnx_asr.load_vad(request.param)


def test_recognize_with_vad(model: TextResultsAsrAdapter, vad: BaseVad) -> None:
    rng = np.random.default_rng(0)
    waveform = rng.random((10 * 16_000), dtype=np.float32)

    result = model.with_vad(vad).recognize(waveform)
    assert isinstance(result, Iterator)
    assert all(isinstance(item, SegmentResult) for item in result)


def test_recognize_with_vad_and_timestamps(model: TextResultsAsrAdapter, vad: BaseVad) -> None:
    rng = np.random.default_rng(0)
    waveform = rng.random((10 * 16_000), dtype=np.float32)

    result = model.with_vad(vad).with_timestamps().recognize(waveform)
    assert isinstance(result, Iterator)
    assert all(isinstance(item, TimestampedSegmentResult) for item in result)
