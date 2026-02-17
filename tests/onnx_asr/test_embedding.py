import numpy as np
import pytest

from onnx_asr.adapters import SeAdapter
from onnx_asr.loader import Manager


@pytest.fixture(scope="module", params=["wespeaker/wespeaker-voxceleb-resnet34"])
def model(request: pytest.FixtureRequest) -> SeAdapter:
    manager = Manager()
    return manager.create_se(request.param)


def test_embedding(model: SeAdapter) -> None:
    rng = np.random.default_rng(0)
    waveform = rng.random((1 * 16_000), dtype=np.float32)

    result = model.embedding(waveform)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.ndim == 1


def test_empty_embedding(model: SeAdapter) -> None:
    result = model.embedding([])
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.ndim == 0


def test_embedding_batch(model: SeAdapter) -> None:
    rng = np.random.default_rng(0)
    waveform1 = rng.random((2 * 16_000), dtype=np.float32)
    waveform2 = rng.random((1 * 16_000), dtype=np.float32)

    result = model.embedding([waveform1, waveform2])
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.ndim == 2
    assert result.shape[0] == 2
