import numpy as np
import pytest


def create_waveforms():
    rng = np.random.default_rng(0)
    return [rng.random((16_000 * 5 + x), dtype=np.float32) * 2 - 1 for x in [0, 79, 80, -1, -10000]]


@pytest.fixture(scope="session")
def waveform_batch():
    return create_waveforms()


def pytest_generate_tests(metafunc):
    if "waveforms" in metafunc.fixturenames:
        batch = create_waveforms()
        metafunc.parametrize("waveforms", [waveform.reshape(1, -1) for waveform in batch] + [batch])
