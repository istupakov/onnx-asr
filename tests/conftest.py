import numpy as np
import pytest


def create_waveforms(base_sec: int):
    rng = np.random.default_rng(0)
    return [rng.random((base_sec * 16_000 + x), dtype=np.float32) * 2 - 1 for x in [0, 1, 79, 80, -1, -10000]]


def pytest_generate_tests(metafunc: pytest.Metafunc):
    if "waveforms" in metafunc.fixturenames:
        batch = create_waveforms(30 if "whisper" in metafunc.module.__name__ else 5)
        metafunc.parametrize("waveforms", [waveform.reshape(1, -1) for waveform in batch] + [batch])
