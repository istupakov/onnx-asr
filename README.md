# ONNX ASR

[![PyPI - Version](https://img.shields.io/pypi/v/onnx-asr)](https://pypi.org/project/onnx-asr)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/onnx-asr)](https://pypi.org/project/onnx-asr)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/onnx-asr)](https://pypi.org/project/onnx-asr)
[![PyPI - Types](https://img.shields.io/pypi/types/onnx-asr)](https://pypi.org/project/onnx-asr)
[![PyPI - License](https://img.shields.io/pypi/l/onnx-asr)](https://github.com/istupakov/onnx-asr/blob/main/LICENSE)<br>
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![mypy](https://img.shields.io/badge/mypy-checked-blue)](https://mypy-lang.org/)
[![Material for MkDocs](https://img.shields.io/badge/Material_for_MkDocs-526CFE?logo=MaterialForMkDocs&logoColor=white)](https://squidfunk.github.io/mkdocs-material/)
[![CodeFactor Grade](https://img.shields.io/codefactor/grade/github/istupakov/onnx-asr)](https://www.codefactor.io/repository/github/istupakov/onnx-asr/overview/main)
[![Codecov](https://img.shields.io/codecov/c/github/istupakov/onnx-asr)](https://codecov.io/github/istupakov/onnx-asr)
[![GitHub - CI](https://github.com/istupakov/onnx-asr/actions/workflows/python-package.yml/badge.svg)](https://github.com/istupakov/onnx-asr/actions/workflows/python-package.yml)

**onnx-asr** is a Python package for Automatic Speech Recognition using ONNX models. It's a lightweight, fast, and easy-to-use pure Python package with minimal dependencies (no need for PyTorch, Transformers, or FFmpeg):

[![numpy](https://img.shields.io/badge/numpy-required-blue?logo=numpy)](https://pypi.org/project/numpy/)
[![onnxruntime](https://img.shields.io/badge/onnxruntime-required-blue?logo=onnx)](https://pypi.org/project/onnxruntime/)
[![huggingface-hub](https://img.shields.io/badge/huggingface--hub-optional-blue?logo=huggingface)](https://pypi.org/project/huggingface-hub/)

Key features of **onnx-asr** include:

* Supports many modern ASR [models](https://istupakov.github.io/onnx-asr/usage/#supported-model-names)
* Runs on a wide range of devices, from small IoT / edge devices to servers with powerful GPUs ([benchmarks](https://istupakov.github.io/onnx-asr/benchmarks/))
* Works on Windows, Linux, and macOS on x86 and Arm CPUs, with support for CUDA, TensorRT, CoreML, ROCm, and DirectML
* Supports NumPy versions from 1.22 to 2.4+ and Python versions from 3.10 to 3.14
* Loads models from Hugging Face or local directories, including quantized versions
* Accepts WAV files or NumPy arrays, with built-in file reading and resampling
* Supports custom models (if their architecture is supported)
* Supports batch processing
* Supports long-form recognition using [VAD](https://istupakov.github.io/onnx-asr/usage/#vad-voice-activity-detection) (Voice Activity Detection)
* Can return token-level timestamps and log probabilities
* Provides a fully typed and well-documented [Python API](https://istupakov.github.io/onnx-asr/reference/)
* Provides a simple command-line interface ([CLI](https://istupakov.github.io/onnx-asr/usage/#cli))

> [!NOTE]
> Supports **Parakeet v2 (En) / v3 (Multilingual)**, **Canary v2 (Multilingual)** and **GigaAM v2/v3 (Ru)** models!

> [!WARNING]
> Onnxruntime 1.24.1 does not support symlinks to data files used in the HuggingFace cache for large models. Please upgrade to 1.24.2!

> [!TIP]
> You can check onnx-asr demo on HF Spaces:
> 
> [![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-xl-dark.svg)](https://istupakov-onnx-asr.hf.space/)


## Quickstart

Install onnx-asr:
```sh
pip install onnx-asr[cpu,hub]
```

Load model and recognize WAV file:
```py
import onnx_asr

# Load the Parakeet TDT v3 model from Hugging Face (may take a few minutes)
model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3")

# Recognize speech and print result
result = model.recognize("test.wav")
print(result)
```

> [!WARNING]
> The maximum audio length for most models is 20-30 seconds. For longer audio, [VAD](https://istupakov.github.io/onnx-asr/usage/#vad-voice-activity-detection) can be used.

For more examples, see the [Usage Guide](https://istupakov.github.io/onnx-asr/usage/).

## Supported Model Architectures

The package supports the following modern ASR model architectures (see the [supported model names](https://istupakov.github.io/onnx-asr/usage/#supported-model-names) for the full list of models and [comparison](https://istupakov.github.io/onnx-asr/comparison/) with original implementations):

* Nvidia NeMo Conformer/FastConformer/Parakeet/Canary (with CTC, RNN-T, TDT and Transformer decoders)
* Kaldi Icefall Zipformer (with stateless RNN-T decoder) including Alpha Cephei Vosk 0.52+
* GigaChat GigaAM v2/v3 (with CTC and RNN-T decoders, including E2E versions)
* T-Tech T-one (with CTC decoder, no streaming support yet)
* OpenAI Whisper

When saving these models in ONNX format, usually only the encoder and decoder are saved. To run them, the corresponding preprocessor and decoding must be implemented. Therefore, the package contains these implementations for all supported models:

* Log-mel spectrogram preprocessors
* Greedy search decoding

## Installation

See the [Installation Guide](https://istupakov.github.io/onnx-asr/installation/) for detailed instructions.

## Usage Examples

See the [Usage Guide](https://istupakov.github.io/onnx-asr/usage/) for detailed examples.

## Troubleshooting / FAQ

See the [Troubleshooting Guide](https://istupakov.github.io/onnx-asr/troubleshooting/) for common issues and solutions.

For more help, check the [GitHub Issues](https://github.com/istupakov/onnx-asr/issues) or open a new one.

## Benchmarks

See the [Benchmarks](https://istupakov.github.io/onnx-asr/benchmarks/) page for detailed performance benchmarks.

## Comparison with Original Implementations

See the [Comparison Guide](https://istupakov.github.io/onnx-asr/comparison/) for detailed performance comparisons with original implementations.


## Convert Model to ONNX

See the [Conversion Guide](https://istupakov.github.io/onnx-asr/conversion/) for instructions on converting models to ONNX format.

## License

[MIT License](https://github.com/istupakov/onnx-asr/blob/main/LICENSE)
