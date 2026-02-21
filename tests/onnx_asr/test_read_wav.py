from pathlib import Path
from typing import Literal, get_args

import numpy as np
import pytest

try:
    import soundfile as sf
except (ImportError, OSError):
    pytest.skip("soundfile not available", allow_module_level=True)

from onnx_asr.utils import (
    DifferentSampleRatesError,
    SampleRates,
    SupportedOnlyMonoAudioError,
    WrongSampleRateError,
    read_wav,
    read_wav_files,
)


@pytest.mark.parametrize("subtype", ["PCM_16", "PCM_24", "PCM_32", "PCM_U8"])
@pytest.mark.parametrize("sample_rate", [16_000, 44_100])
@pytest.mark.parametrize("channels", [1, 2])
def test_read_wav(tmp_path: Path, subtype: str, sample_rate: int, channels: int) -> None:
    rng = np.random.default_rng(0)
    file = tmp_path.joinpath("test.wav")
    sf.write(file, rng.random((1 * sample_rate, channels), dtype=np.float32), sample_rate, subtype)

    expected_data, _ = sf.read(file, dtype="float32", always_2d=True)
    readed_data, readed_sample_rate = read_wav(str(file))

    assert readed_sample_rate == sample_rate
    assert readed_data.shape[1] == channels
    np.testing.assert_equal(readed_data, expected_data)


@pytest.mark.parametrize("sample_rate", get_args(SampleRates))
def test_read_wav_files(tmp_path: Path, sample_rate: SampleRates) -> None:
    rng = np.random.default_rng(0)
    files = [tmp_path.joinpath(f"test{k}.wav") for k in [1, 2]]
    data = [rng.random((k * sample_rate), dtype=np.float32) for k in [1, 2, 3, 4]]

    sf.write(files[0], data[0], sample_rate, "PCM_32")
    sf.write(files[1], data[1], sample_rate, "PCM_32")

    result_data, result_lens, result_sample_rate = read_wav_files(
        [*files, data[2].reshape(1, -1), data[3].reshape(-1, 1)], numpy_sample_rate=sample_rate
    )
    assert result_sample_rate == sample_rate
    assert result_data.shape[0] == len(data)
    np.testing.assert_equal(result_lens, [d.shape[0] for d in data])
    for i in range(len(data)):
        np.testing.assert_equal(result_data[i], np.pad(data[i], (0, result_data.shape[1] - result_lens[i])))


@pytest.mark.parametrize("sample_rate", [16_000, 44_100])
@pytest.mark.parametrize("channel", [0, 1, "mean"])
def test_read_wav_files_multichannel(tmp_path: Path, sample_rate: SampleRates, channel: int | Literal["mean"]) -> None:
    rng = np.random.default_rng(0)
    files = [tmp_path.joinpath(f"test{k}.wav") for k in [1, 2]]
    data = [rng.random((k * sample_rate, 2), dtype=np.float32) for k in [1, 2, 3, 4]]

    sf.write(files[0], data[0], sample_rate, "PCM_32")
    sf.write(files[1], data[1], sample_rate, "PCM_32")

    result_data, result_lens, result_sample_rate = read_wav_files(
        [*files, data[2], data[3]], numpy_sample_rate=sample_rate, channel=channel
    )
    assert result_sample_rate == sample_rate
    assert result_data.shape[0] == len(data)
    np.testing.assert_equal(result_lens, [d.shape[0] for d in data])
    for i in range(len(data)):
        expected = data[i].mean(axis=-1) if channel == "mean" else data[i][:, channel]
        np.testing.assert_equal(result_data[i], np.pad(expected, (0, result_data.shape[1] - result_lens[i])))


def test_read_wav_files_mono_error_numpy() -> None:
    rng = np.random.default_rng(0)
    data = rng.random((16_000, 2), dtype=np.float32)

    with pytest.raises(SupportedOnlyMonoAudioError):
        read_wav_files([data], 16_000)


def test_read_wav_files_mono_error_file(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    file = tmp_path.joinpath("test.wav")
    data = rng.random((16_000, 2), dtype=np.float32)
    sf.write(file, data, 16_000)

    with pytest.raises(SupportedOnlyMonoAudioError):
        read_wav_files([file], 16_000)


def test_read_wav_files_different_sample_rates_error(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    file = tmp_path.joinpath("test.wav")
    data = rng.random((16_000), dtype=np.float32)
    sf.write(file, data, 16_000)

    with pytest.raises(DifferentSampleRatesError):
        read_wav_files([file, data], 8_000)


def test_read_wav_files_wrong_sample_rate_error(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    file = tmp_path.joinpath("test.wav")
    sf.write(file, rng.random((16_000), dtype=np.float32), 10_000)

    with pytest.raises(WrongSampleRateError):
        read_wav_files([file], 16_000)
