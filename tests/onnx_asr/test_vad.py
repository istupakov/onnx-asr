from collections.abc import Iterator
from typing import get_args

import numpy as np
import pytest

import onnx_asr
from onnx_asr.adapters import TextResultsAsrAdapter
from onnx_asr.loader import VadNames
from onnx_asr.vad import BaseVad, SegmentResult, TimestampedSegmentResult, Vad


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
