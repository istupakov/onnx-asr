import numpy as np
import pytest


@pytest.fixture
def base_sec():
    return 5


@pytest.fixture(params=["single", "batch"])
def waveforms(base_sec: int, request: pytest.FixtureRequest) -> list[np.ndarray]:
    rng = np.random.default_rng(0)

    if request.param == "single":
        return [rng.random((base_sec * 16_000), dtype=np.float32) * 2 - 1]

    return [rng.random((base_sec * 16_000 + x), dtype=np.float32) * 2 - 1 for x in [0, 1, 79, 80, -1, -10000]]
