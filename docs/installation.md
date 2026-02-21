# Installation

The package can be installed from [PyPI](https://pypi.org/project/onnx-asr/).

## Quick Install

```sh
# CPU only / CoreML for Apple Silicon (simplest install, minimal native deps)
pip install onnx-asr[cpu,hub]

# With NVIDIA GPU support (requires installed CUDA/cuDNN/TensorRT)
pip install onnx-asr[gpu,hub]

# Using uv
uv pip install onnx-asr[cpu,hub]
```

## Requirements

### ONNX Runtime Packages

onnx-asr requires an ONNX Runtime package. There are several options depending on your OS and hardware:

| Package | Providers | Notes |
|---------|-----------|-------|
| `onnxruntime` | `CPUExecutionProvider`, `CoreMLExecutionProvider` | Default, works on all platforms |
| `onnxruntime-gpu` | `CPUExecutionProvider`, `CUDAExecutionProvider`, `TensorrtExecutionProvider` | For NVIDIA GPUs (requires NVIDIA libs) |
| `onnxruntime-directml`, `onnxruntime-windowsml` (new) | `CPUExecutionProvider`, `DmlExecutionProvider` | DirectML - only for Windows but no additional deps |
| `onnxruntime-webgpu` (beta) | `CPUExecutionProvider`, `WebGpuExecutionProvider` | WebGPU - cross-platform and no additional deps |

Additional ONNX Runtime packages (not tested):

- `onnxruntime-trt-rtx` - NVIDIA TensorRT for RTX
- `onnxruntime-qnn` - Qualcomm Snapdragon
- `onnxruntime-openvino` - Intel OpenVINO
- `onnxruntime-rocm` - AMD GPUs (legacy)
- `onnxruntime-migraphx` - AMD GPUs (new)
- `onnxruntime-cann` - Huawei Ascend NPU

> [!NOTE]
> Only `onnxruntime` and `onnxruntime-gpu` have predefined extras (`[cpu]` and `[gpu]`). Other packages can be installed manually.

### Supported Platforms

The supported platforms are primarily determined by available ONNX Runtime wheels.

| OS / CPU | Python 3.10 | Python 3.11 | Python 3.12 | Python 3.13 | Python 3.14 | Python 3.13t | Python 3.14t |
|----------|-------------|-------------|-------------|-------------|-------------|--------------|--------------|
| Linux x86_64 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Linux Arm64 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Windows x86_64 | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| Windows Arm64 | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| macOS Arm64 (Apple Silicon) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| macOS x86_64 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |

### Minimum Dependency Versions

| Package | Minimum Version |
|---------|-----------------|
| numpy | 1.22.4 |
| onnxruntime | 1.18.1 (or any ONNX Runtime package) |
| huggingface-hub | 0.30.2 (optional, for model downloading) |
| typing-extensions | 4.6.0 (Python < 3.11 only) |

> [!NOTE]
> Older versions of ONNX Runtime packages may work but are not tested. The minimum version listed is the one used in CI testing.

## Install from PyPI

1. With CPU `onnxruntime` and `huggingface-hub`:

    ```sh
    pip install onnx-asr[cpu,hub]
    ```

2. With `onnxruntime` for NVIDIA GPUs and `huggingface-hub`:

    ```sh
    pip install onnx-asr[gpu,hub]
    ```

    > [!WARNING]
    > First, you need to install the [required](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements) version of CUDA/cuDNN for `CUDAExecutionProvider` and [required](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#requirements) TensorRT for `TensorrtExecutionProvider` (optional).

    You can also install `onnxruntime` CUDA/cuDNN dependencies and TensorRT via pip:

    ```sh
    pip install onnxruntime-gpu[cuda,cudnn] tensorrt-cu12-libs
    ```

3. With `onnxruntime` for [WinML](https://onnxruntime.ai/docs/get-started/with-windows.html) and `huggingface-hub`:

    ```sh 
    pip install onnx-asr[hub] onnxruntime-windowsml
    ```

4. Without `onnxruntime` and `huggingface-hub` (if you already have some version of `onnxruntime` installed and prefer to download the models yourself):

    ```sh
    pip install onnx-asr
    ```

## Install from source

To install the latest version of `onnx-asr` from sources, use `pip` (or `uv pip`):

```sh
pip install git+https://github.com/istupakov/onnx-asr
```
