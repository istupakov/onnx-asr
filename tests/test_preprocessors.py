import pytest
import numpy as np

import preprocessors
from onnx_asr import Preprocessor


@pytest.mark.parametrize(
    "preprocessor, equal",
    [
        pytest.param(preprocessors.gigaam_preprocessor_torch, True, id="torch"),
        pytest.param(preprocessors.GigaamPreprocessor, False, id="onnx_func"),
        pytest.param(Preprocessor("gigaam"), False, id="onnx_model"),
    ],
)
def test_gigaam_preprocessor(preprocessor, equal, waveforms):
    waveforms, lens = preprocessors.pad_list(waveforms)
    expected, expected_lens = preprocessors.gigaam_preprocessor_origin(waveforms, lens)
    actual, actual_lens = preprocessor(waveforms, lens)

    assert expected.shape[2] == max(expected_lens)
    np.testing.assert_equal(actual_lens, expected_lens)
    if equal:
        np.testing.assert_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, atol=5e-5)


@pytest.mark.parametrize(
    "preprocessor",
    [
        pytest.param(preprocessors.kaldi_preprocessor_torch, id="torch"),
        pytest.param(preprocessors.KaldiPreprocessor, id="onnx_func"),
        pytest.param(Preprocessor("kaldi"), id="onnx_model"),
    ],
)
def test_kaldi_preprocessor(preprocessor, waveforms):
    waveforms, lens = preprocessors.pad_list(waveforms)
    expected, expected_lens = preprocessors.kaldi_preprocessor_origin(waveforms, lens)
    actual, actual_lens = preprocessor(waveforms, lens)

    assert expected.shape[1] == max(expected_lens)
    np.testing.assert_equal(actual_lens, expected_lens)
    np.testing.assert_allclose(actual, expected, atol=5e-4)


@pytest.mark.parametrize(
    "preprocessor, atol",
    [
        pytest.param(preprocessors.nemo_preprocessor_torch, 5e-5, id="torch"),
        pytest.param(preprocessors.NemoPreprocessor, 1e-4, id="onnx_func"),
        pytest.param(Preprocessor("nemo"), 1e-4, id="onnx_model"),
    ],
)
def test_nemo_preprocessor(preprocessor, atol, waveforms):
    waveforms, lens = preprocessors.pad_list(waveforms)
    expected, expected_lens = preprocessors.nemo_preprocessor_origin(waveforms, lens)
    actual, actual_lens = preprocessor(waveforms, lens)

    assert expected.shape[2] == max(expected_lens)
    np.testing.assert_equal(actual_lens, expected_lens)
    np.testing.assert_allclose(actual, expected, atol=atol)


@pytest.mark.parametrize(
    "preprocessor, equal",
    [
        pytest.param(preprocessors.whisper_preprocessor_torch, True, id="torch"),
        pytest.param(preprocessors.WhisperPreprocessor, False, id="onnx_func"),
        pytest.param(Preprocessor("whisper"), False, id="onnx_model"),
    ],
)
def test_whisper_preprocessor(preprocessor, equal, waveforms):
    waveforms, lens = preprocessors.pad_list(waveforms)
    expected, expected_lens = preprocessors.whisper_preprocessor_origin(waveforms, lens)
    actual, actual_lens = preprocessor(waveforms, lens)

    assert expected.shape[2] == max(expected_lens)
    np.testing.assert_equal(actual_lens, expected_lens)
    if equal:
        np.testing.assert_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, atol=1e-5)
