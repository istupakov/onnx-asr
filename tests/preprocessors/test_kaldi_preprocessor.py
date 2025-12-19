import kaldi_native_fbank as knf
import numpy as np
import pytest
import torch
import torchaudio

from onnx_asr.preprocessors import Preprocessor
from onnx_asr.utils import pad_list
from preprocessors import kaldi


def pad_features(arrays):
    lens = np.array([array.shape[0] for array in arrays])
    max_len = lens.max()
    return np.stack([np.pad(array, ((0, max_len - array.shape[0]), (0, 0))) for array in arrays]), lens


def preprocessor_origin(waveforms, lens, remove_dc_offset=kaldi.remove_dc_offset):
    opts = knf.FbankOptions()
    opts.frame_opts.snip_edges = kaldi.snip_edges
    opts.frame_opts.dither = kaldi.dither
    opts.frame_opts.remove_dc_offset = remove_dc_offset
    opts.frame_opts.preemph_coeff = kaldi.preemphasis_coefficient
    opts.mel_opts.num_bins = kaldi.num_mel_bins
    opts.mel_opts.high_freq = kaldi.high_freq

    results = []
    for waveform, len in zip(waveforms, lens, strict=True):
        fbank = knf.OnlineFbank(opts)
        fbank.accept_waveform(kaldi.sample_rate, waveform[:len])
        fbank.input_finished()
        results.append(np.array([fbank.get_frame(i) for i in range(fbank.num_frames_ready)]))

    return pad_features(results)


def preprocessor_torch(waveforms, lens):
    results = []
    for waveform, len in zip(waveforms, lens, strict=True):
        results.append(
            torchaudio.compliance.kaldi.fbank(
                torch.from_numpy(waveform[:len]).unsqueeze(0).contiguous(),
                snip_edges=kaldi.snip_edges,
                dither=kaldi.dither,
                remove_dc_offset=kaldi.remove_dc_offset,
                preemphasis_coefficient=kaldi.preemphasis_coefficient,
                num_mel_bins=kaldi.num_mel_bins,
                high_freq=kaldi.high_freq,
            ).numpy()
        )

    return pad_features(results)


@pytest.fixture(scope="module")
def preprocessor(request):
    match request.param:
        case "torch":
            return preprocessor_torch
        case "onnx_func":
            return kaldi.KaldiPreprocessor
        case "onnx_func_fast":
            return kaldi.KaldiPreprocessorFast
        case "onnx_model":
            return Preprocessor("kaldi", {})
        case "onnx_model_mt":
            return Preprocessor("kaldi", {"max_concurrent_workers": 2})
        case "onnx_model_fast":
            return Preprocessor("kaldi_fast", {})
        case "onnx_model_fast_mt":
            return Preprocessor("kaldi_fast", {"max_concurrent_workers": 2})


@pytest.mark.parametrize("preprocessor", ["torch", "onnx_func", "onnx_model", "onnx_model_mt"], indirect=["preprocessor"])
def test_kaldi_preprocessor(preprocessor, waveforms):
    waveforms, lens = pad_list(waveforms)
    expected, expected_lens = preprocessor_origin(waveforms, lens)
    actual, actual_lens = preprocessor(waveforms, lens)

    assert expected.shape[1] == max(expected_lens)
    np.testing.assert_equal(actual_lens, expected_lens)
    np.testing.assert_allclose(actual, expected, atol=5e-4, rtol=1e-4)


@pytest.mark.parametrize("preprocessor", ["onnx_func_fast", "onnx_model_fast", "onnx_model_fast_mt"], indirect=["preprocessor"])
def test_kaldi_preprocessor_fast(preprocessor, waveforms):
    waveforms, lens = pad_list(waveforms)
    expected, expected_lens = preprocessor_origin(
        waveforms - waveforms.mean(axis=-1, keepdims=True), lens, remove_dc_offset=False
    )
    actual, actual_lens = preprocessor(waveforms, lens)

    assert expected.shape[1] == max(expected_lens)
    np.testing.assert_equal(actual_lens, expected_lens)
    np.testing.assert_allclose(actual, expected, atol=5e-4, rtol=1e-4)
