from onnx_asr.loader import Manager


def test_with_cpu_provider() -> None:
    providers = ["CPUExecutionProvider"]
    manager = Manager(providers=providers)

    assert manager.default_onnx_config.get("providers") == providers
    assert manager.preprocessor_max_workers == 1
    assert manager.use_numpy_preprocessors is True
    assert manager.preprocessor_config.get("providers") == providers
    assert manager.resampler_config.get("providers") == providers


def test_with_cuda_provider() -> None:
    providers = ["CUDAExecutionProvider"]
    manager = Manager(providers=providers)

    assert manager.default_onnx_config.get("providers") == providers
    assert manager.preprocessor_max_workers == 1
    assert manager.use_numpy_preprocessors is True
    assert manager.preprocessor_config.get("providers") == []
    assert manager.resampler_config.get("providers") == providers


def test_with_tensorrt_provider() -> None:
    providers = ["TensorrtExecutionProvider"]
    manager = Manager(providers=providers)

    assert manager.default_onnx_config.get("providers") == providers
    assert manager.preprocessor_max_workers == 1
    assert manager.use_numpy_preprocessors is False
    assert manager.preprocessor_config.get("providers") == providers
    assert manager.resampler_config.get("providers") == []
